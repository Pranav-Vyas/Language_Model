"""Wikipedia scraper for one language tree (Model-H / Model-L manual tier).

Duplicated verbatim into each language's ``data_prep/`` — it never imports from a
sibling tree. The module auto-loads *its own* tree's config (the single YAML in
``../configs/``) so the identical file behaves correctly in ``hindi/`` and
``nepali/`` without edits.

Pipeline (each step writes to disk before the next begins, so a kill is cheap):
  1. enumerate_titles  -> corpus/raw/wiki_<lang>.titles.jsonl   (all article titles)
  2. fetch_all         -> corpus/raw/wiki_<lang>.docs.jsonl      (clean plain text)
                       -> corpus/raw/manifest.jsonl             (one provenance row/doc)
                       -> corpus/raw/wiki_<lang>.flagged.jsonl   (over-short = review)

Resumability: "done" is defined by the manifest. On restart we skip every title
already recorded there, so re-running does zero duplicate fetches.

Etiquette: descriptive User-Agent, a conservative min interval between requests,
``maxlag=5``, and honouring HTTP 429 / Retry-After with backoff.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mwparserfromhell
import requests

# This tree's root (…/hindi or …/nepali) — used to find config + write outputs.
TREE = Path(__file__).resolve().parent.parent

# Contact string embedded in the User-Agent, per MediaWiki API etiquette.
CONTACT = "anuragkaushal101@gmail.com"

# Wikitext tags whose *contents* are non-prose and must be removed entirely.
DROP_TAGS = {
    "ref", "references", "gallery", "table", "math", "timeline",
    "imagemap", "score", "syntaxhighlight", "source", "poem", "mapframe",
}


# --------------------------------------------------------------------------- #
# Config loading (dynamic, so the identical file works in either tree)
# --------------------------------------------------------------------------- #
def _load_tree_config():
    """Import this tree's own ``config.py`` and load its single YAML config."""
    cfg_py = TREE / "config.py"
    spec = importlib.util.spec_from_file_location(f"{TREE.name}_config", cfg_py)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution can find the module.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yamls = sorted((TREE / "configs").glob("*.yaml"))
    if len(yamls) != 1:
        raise RuntimeError(f"expected exactly one config yaml in {TREE/'configs'}, found {yamls}")
    return mod.load_config(yamls[0])


# --------------------------------------------------------------------------- #
# Markup stripping: wikitext -> clean plain text
# --------------------------------------------------------------------------- #
# Tables in wikitext are `{| ... |}` blocks that mwparserfromhell does not always
# model as removable nodes; strip them first with a (non-greedy, multiline) regex.
_TABLE_RE = re.compile(r"\{\|.*?\|\}", re.DOTALL)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_MULTINL_RE = re.compile(r"\n{3,}")
# A leftover line that is pure table/markup residue (starts with |, !, {, }).
_RESIDUE_LINE_RE = re.compile(r"^\s*[|!{}]+")
# Language-code-like interwiki prefixes, e.g. [[en:Foo]], [[fr:...]].
_LANGCODE_RE = re.compile(r"^[a-z]{2,3}(-[a-z]+)?$")

# --- residue sweep for MALFORMED source wikitext ---------------------------
# Small wikis contain unbalanced/broken markup that mwparserfromhell cannot
# parse into removable nodes, so it falls through strip_code() as literal text.
# Found at full-corpus scale on a small Devanagari-script wiki (~0.4% of articles);
# each pattern below is a real example pulled from the affected articles, not
# a hypothetical. These run AFTER strip_code to mop up what the parser passed
# through, keeping human-readable content and dropping only markup debris.
#
#   `[[Target|Display]`        -- one closing bracket instead of two
#   `<ref name=x>< /ref>`      -- space inside the closing tag
#   `{{}}`                     -- empty template call; has no name, so
#                                  mwparserfromhell never sees it as a Template
#   `</small>` with no opener  -- an orphaned closing tag (editor left one half
#                                  behind); strip_code only removes matched pairs
#   `<ref name="x" | title=…`  -- malformed attribute syntax (stray `|`) that
#                                  breaks tag parsing, so the whole thing stays
#                                  literal, `>` and all
_REF_FRAGMENT_RE = re.compile(r"<\s*/?\s*ref[^>]*>", re.IGNORECASE)
_BROKEN_LINK_RE = re.compile(r"\[\[([^\[\]]*?)\]{1,2}")
_STRAY_BRACKETS_RE = re.compile(r"\[\[|\]\]")
# A tag that DOES close within the same line, e.g. `<ref name="x" | title=y>`.
# Tag name is NOT restricted to ASCII: small wikis contain localized pseudo-tags
# like `<संदर्भ>...</संदर्भ>` (Devanagari for "reference", used in place of <ref>)
# that neither MediaWiki nor mwparserfromhell recognise as real tags, so they
# survive as literal text. `[^\s<>/]` accepts any script for the tag name.
_CLOSED_TAG_RE = re.compile(r"</?[^\s<>/][^<>\n]{0,300}>")
# A tag with NO closing `>` before the line ends -- strip from `<` to end of
# line only, so we can never eat into the next paragraph's real content.
_UNCLOSED_TAG_RE = re.compile(r"<[^\s<>/][^<>\n]*$")


def _strip_balanced_template_debris(text: str) -> str:
    """Remove leftover `{{ ... }}` template debris of ANY nesting depth.

    Genuine templates are already removed via `filter_templates()` upstream;
    if `{{` still appears at this point it means that removal failed -- most
    often because the template's own parameters contain further malformed
    markup that broke mwparserfromhell's parser (observed: `{{Efn|...}}`
    footnotes whose content includes other templates/refs). Nesting depth is
    unbounded in principle, so this is a small hand-written scanner rather
    than a regex: track `{{`/`}}` depth and only keep characters seen at
    depth 0. An unbalanced (never-closed) `{{` at end-of-text is handled for
    free -- everything from it onward is already being skipped.
    """
    out = []
    depth = 0
    i, n = 0, len(text)
    while i < n:
        if text[i:i+2] == "{{":
            depth += 1
            i += 2
            continue
        if text[i:i+2] == "}}" and depth > 0:
            depth -= 1
            i += 2
            continue
        if depth == 0:
            out.append(text[i])
        i += 1
    return "".join(out)


def _sweep_markup_residue(text: str, drop_link_prefixes: set[str]) -> str:
    """Remove markup fragments left behind by unbalanced/malformed wikitext.

    Keeps the display half of a broken link (`[[Target|Display]` -> `Display`)
    so we recover the real words instead of deleting the sentence. Runs
    line-by-line for the tag sweep so an unclosed tag can only ever consume
    the rest of its own line, never subsequent paragraphs.

    `drop_link_prefixes` also catches SINGLE-bracket namespace links, e.g.
    `[श्रेणी:...]]` (a source typo missing one `[`) that mwparserfromhell's
    wikilink parser -- which requires `[[...]]` -- never recognises as a link
    node at all, so it isn't touched by the normal namespace-drop logic. And,
    as a final generic net, ANY remaining bare `<prefix>:` occurrence -- with
    or without brackets -- since a category/file namespace prefix followed by
    a colon cannot occur in genuine prose; the one observed source was a
    wikilink nested inside another wikilink's title (`[[श्रेणी:[[श्रेणी:]]]]`),
    which mwparserfromhell cannot remove because the inner node lives inside
    the outer node's title attribute, not the document tree it can edit.
    """
    text = _REF_FRAGMENT_RE.sub("", text)
    text = _BROKEN_LINK_RE.sub(lambda m: m.group(1).split("|")[-1], text)
    if drop_link_prefixes:
        alt = "|".join(re.escape(p) for p in drop_link_prefixes)
        single_bracket_ns_re = re.compile(rf"\[\s*(?:{alt})\s*:[^\[\]\n]{{0,120}}\]{{1,2}}", re.IGNORECASE)
        text = single_bracket_ns_re.sub("", text)
    text = _STRAY_BRACKETS_RE.sub("", text)
    text = _strip_balanced_template_debris(text)
    lines = []
    for line in text.split("\n"):
        line = _CLOSED_TAG_RE.sub("", line)
        line = _UNCLOSED_TAG_RE.sub("", line)
        lines.append(line)
    text = "\n".join(lines)
    if drop_link_prefixes:
        alt = "|".join(re.escape(p) for p in drop_link_prefixes)
        # A plain \b boundary is NOT safe here: Python's \b fires at any \w /
        # non-\w transition, and Devanagari combining marks (matras, virama)
        # are inconsistently classified as \w, so \b can spuriously fire
        # *inside* a word -- verified this would wrongly turn "प्रोफाइल:"
        # (profile, containing "फाइल" as a substring) into "प्रोईनशा". Instead,
        # require the preceding character not be Devanagari at all (U+0900-
        # U+097F): that blocks the mid-word case while still matching a real
        # leaked prefix preceded by whitespace, a bracket, or start-of-string.
        bare_ns_re = re.compile(rf"(?<![ऀ-ॿ])(?:{alt})\s*:", re.IGNORECASE)
        text = bare_ns_re.sub("", text)
    return text


def strip_wikitext(wikitext: str, drop_link_prefixes: set[str]) -> str:
    """Convert raw wikitext to clean plain-text paragraphs.

    Removes templates/infoboxes, ref/table/gallery tags (with contents),
    namespace links (Category/File/…, incl. localized names) and interwiki
    language links, external-link URLs, and leftover table residue — while
    KEEPING the visible display text of ordinary wikilinks.

    `drop_link_prefixes` is the lower-cased set of namespace prefixes to drop,
    built from the wiki's own siteinfo so it works for any language.
    """
    # 1. Kill HTML comments and raw `{| |}` tables before parsing.
    text = _COMMENT_RE.sub("", wikitext)
    text = _TABLE_RE.sub("", text)

    code = mwparserfromhell.parse(text)

    # 2. Remove all templates (infoboxes, cite, navboxes, …).
    for tmpl in list(code.filter_templates(recursive=True)):
        try:
            code.remove(tmpl)
        except ValueError:
            pass  # already removed as part of a parent node

    # 3. Remove non-prose tags together with their contents.
    for tag in list(code.filter_tags(recursive=True)):
        try:
            if str(tag.tag).lower() in DROP_TAGS:
                code.remove(tag)
        except ValueError:
            pass

    # 4. Wikilinks: drop namespace/interwiki links; keep display text otherwise.
    for link in list(code.filter_wikilinks(recursive=True)):
        title = str(link.title).strip()
        prefix = title.split(":", 1)[0].strip().lstrip(":").lower() if ":" in title else ""
        is_namespace = prefix in drop_link_prefixes
        is_interwiki = bool(prefix) and link.text is None and _LANGCODE_RE.match(prefix) is not None
        try:
            if is_namespace or is_interwiki:
                code.remove(link)
            else:
                code.replace(link, str(link.text) if link.text else title)
        except ValueError:
            pass

    # 5. External links -> keep their anchor text if any, else drop.
    for ext in list(code.filter_external_links(recursive=True)):
        try:
            code.replace(ext, str(ext.title) if ext.title else "")
        except ValueError:
            pass

    # 6. Collapse remaining markup to text, sweep malformed residue, then tidy.
    plain = code.strip_code(normalize=True, collapse=True)
    plain = _sweep_markup_residue(plain, drop_link_prefixes)
    lines = []
    for raw_line in plain.split("\n"):
        line = raw_line.strip()
        if not line or _RESIDUE_LINE_RE.match(line):
            continue
        lines.append(line)
    plain = "\n".join(lines)
    plain = _MULTINL_RE.sub("\n\n", plain).strip()
    return plain


# Signatures that must NEVER survive a correctly-cleaned document. Checked
# after every clean as a final verification gate (see `detect_residual_markup`)
# rather than assumed: at 15,267-article scale, source wikitext contains
# malformed markup varied enough (nested wikilinks inside a link's own title,
# non-ASCII pseudo-tags, unbalanced templates, ...) that no fixed set of sweep
# rules can be proven to catch every case. A doc that still matches one of
# these gets FLAGGED for review rather than silently entering the corpus with
# visible markup debris, or requiring an unbounded chase of ever-narrower edge
# cases here. Same philosophy as the short-document flag.
_LEAK_CHECK_PATTERNS = {
    "template": re.compile(r"\{\{|\}\}"),
    "table": re.compile(r"\{\||\|\}"),
    "wikilink_bracket": re.compile(r"\[\[|\]\]"),
    "tag": re.compile(r"<[^\s<>/][^<>\n]{0,60}>"),
}


def detect_residual_markup(text: str, drop_link_prefixes: set[str]) -> list[str]:
    """Return the labels of any leak signatures still present after cleaning.

    Empty list = clean. Namespace-prefix residue (e.g. stray `श्रेणी:`) is
    checked separately with the same Devanagari-safe lookbehind used in the
    sweep, so this never flags the legitimate word "प्रोफाइल" (profile).
    """
    found = [label for label, pat in _LEAK_CHECK_PATTERNS.items() if pat.search(text)]
    if drop_link_prefixes:
        alt = "|".join(re.escape(p) for p in drop_link_prefixes)
        ns_re = re.compile(rf"(?<![ऀ-ॿ])(?:{alt})\s*:", re.IGNORECASE)
        if ns_re.search(text):
            found.append("namespace_prefix")
    return found


# --------------------------------------------------------------------------- #
# Scraper
# --------------------------------------------------------------------------- #
@dataclass
class ScrapeStats:
    fetched: int = 0
    skipped_done: int = 0
    flagged_short: int = 0
    flagged_residual_markup: int = 0
    empty: int = 0


class WikiScraper:
    """Enumerate + fetch + clean one Wikipedia language, resumably."""

    def __init__(
        self,
        wiki_lang: str,
        raw_dir: Path,
        manifest_path: Path,
        source_name: str,
        *,
        min_interval: float = 0.5,   # seconds between requests (~2 req/s)
        max_retries: int = 5,
        batch_size: int = 20,        # titles per revision request
        min_chars: int = 200,        # below this -> flag for review, don't drop
        fertility: float = 1.4,      # rough tokens/word for the pre-tokenizer estimate
        chars_per_token: float | None = None,  # if set, overrides fertility with a
                                                # chars/token estimate instead (used for
                                                # Nepali's measured reference value; kept
                                                # optional so Hindi's already-reported
                                                # numbers never silently change)
        timeout: int = 30,
    ) -> None:
        self.api = f"https://{wiki_lang}.wikipedia.org/w/api.php"
        self.wiki_lang = wiki_lang
        self.raw_dir = Path(raw_dir)
        self.manifest_path = Path(manifest_path)
        self.source_name = source_name
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.min_chars = min_chars
        self.fertility = fertility
        self.chars_per_token = chars_per_token
        self.timeout = timeout

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.titles_path = self.raw_dir / f"wiki_{wiki_lang}.titles.jsonl"
        self.docs_path = self.raw_dir / f"wiki_{wiki_lang}.docs.jsonl"
        self.flagged_path = self.raw_dir / f"wiki_{wiki_lang}.flagged.jsonl"

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                f"LMA-IndividualProject/0.1 (academic coursework; monolingual LM corpus; "
                f"contact {CONTACT}) python-requests/{requests.__version__}"
            )
        })
        self._last_request = 0.0
        self._drop_prefixes: set[str] = set()

    # ---- polite HTTP with retry/backoff -------------------------------------
    def _get(self, params: dict) -> dict:
        """One GET to the API with rate-limiting, maxlag, and 429/5xx backoff."""
        params = {**params, "format": "json", "maxlag": 5}
        for attempt in range(self.max_retries):
            # Space requests out to respect the configured rate.
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(self.api, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                self._backoff(attempt, reason=f"network error: {exc}")
                continue
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 5))
                self._sleep(retry_after, reason="HTTP 429 rate limit")
                continue
            if resp.status_code >= 500:
                self._backoff(attempt, reason=f"HTTP {resp.status_code}")
                continue
            data = resp.json()
            # maxlag: the server is lagged; back off as instructed.
            if "error" in data and data["error"].get("code") == "maxlag":
                self._sleep(5, reason="maxlag")
                continue
            return data
        raise RuntimeError(f"API request failed after {self.max_retries} retries: {params}")

    def _backoff(self, attempt: int, reason: str) -> None:
        delay = min(2 ** attempt, 30)
        self._sleep(delay, reason=reason)

    def _sleep(self, seconds: float, reason: str) -> None:
        print(f"  [wait {seconds:.0f}s] {reason}", file=sys.stderr)
        time.sleep(seconds)

    # ---- step 1: enumerate all article titles -------------------------------
    def enumerate_titles(self, force: bool = False) -> int:
        """List every main-namespace article title, paginated, to titles.jsonl.

        Idempotent: if the titles file already exists and `force` is False, we
        keep it (enumeration is a distinct, resumable step).
        """
        if self.titles_path.exists() and not force:
            n = sum(1 for _ in self.titles_path.open(encoding="utf-8"))
            print(f"titles.jsonl already present ({n:,} titles); skipping enumeration.")
            return n
        count = 0
        apcontinue = None
        with self.titles_path.open("w", encoding="utf-8") as out:
            while True:
                params = {
                    "action": "query", "list": "allpages",
                    "apnamespace": 0, "apfilterredir": "nonredirects", "aplimit": 500,
                }
                if apcontinue:
                    params["apcontinue"] = apcontinue
                data = self._get(params)
                for page in data["query"]["allpages"]:
                    out.write(json.dumps(
                        {"page_id": page["pageid"], "title": page["title"]},
                        ensure_ascii=False) + "\n")
                    count += 1
                if "continue" in data:
                    apcontinue = data["continue"]["apcontinue"]
                    print(f"  enumerated {count:,} titles…", file=sys.stderr)
                else:
                    break
        print(f"enumerated {count:,} article titles -> {self.titles_path.name}")
        return count

    # ---- namespace prefixes for the strip step ------------------------------
    def _load_namespace_prefixes(self) -> set[str]:
        """Fetch the wiki's namespace names + aliases (localized) to drop links to
        Category/File/Template/… — this is what makes strip work in any language."""
        data = self._get({"action": "query", "meta": "siteinfo",
                           "siprop": "namespaces|namespacealiases"})
        prefixes: set[str] = set()
        for ns in data["query"]["namespaces"].values():
            if ns["id"] != 0 and ns.get("*"):   # skip main namespace (0)
                prefixes.add(ns["*"].lower())
            if ns.get("canonical"):
                prefixes.add(ns["canonical"].lower())
        for alias in data["query"].get("namespacealiases", []):
            if alias.get("*"):
                prefixes.add(alias["*"].lower())
        return prefixes

    # ---- resumability: which titles are already done ------------------------
    def _done_page_ids(self) -> set[int]:
        """Page ids already recorded in the manifest (the source of truth)."""
        done: set[int] = set()
        if self.manifest_path.exists():
            for line in self.manifest_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a partial trailing line from a hard kill
                if rec.get("source_name") == self.source_name and "page_id" in rec:
                    done.add(rec["page_id"])
        return done

    # ---- step 2+3: fetch latest revision content, clean, record -------------
    def _fetch_revisions(self, titles: list[str]) -> dict[str, dict]:
        """Return {title: {page_id, wikitext}} for a batch of titles (latest rev)."""
        out: dict[str, dict] = {}
        params = {
            "action": "query", "prop": "revisions",
            "rvprop": "content|ids", "rvslots": "main",
            "titles": "|".join(titles), "redirects": 1,
        }
        while True:
            data = self._get(params)
            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                if "missing" in page or "revisions" not in page:
                    continue
                rev = page["revisions"][0]
                content = rev.get("slots", {}).get("main", {}).get("*", "")
                out[page["title"]] = {"page_id": page["pageid"], "wikitext": content}
            if "continue" in data:      # rare: response split across pages
                params.update(data["continue"])
            else:
                break
        return out

    def fetch_all(self, limit: int | None = None, keep_wikitext: bool = False) -> ScrapeStats:
        """Fetch, clean, and record every enumerated title not already done."""
        if not self.titles_path.exists():
            raise RuntimeError("run enumerate_titles() before fetch_all().")
        if not self._drop_prefixes:
            self._drop_prefixes = self._load_namespace_prefixes()

        done = self._done_page_ids()
        print(f"resuming: {len(done):,} already in manifest.")

        # Titles still to do, preserving enumeration order.
        todo: list[tuple[int, str]] = []
        for line in self.titles_path.open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["page_id"] not in done:
                todo.append((rec["page_id"], rec["title"]))
            if limit is not None and len(todo) >= limit:
                break
        print(f"to fetch: {len(todo):,} titles (limit={limit}).")

        stats = ScrapeStats(skipped_done=len(done))
        # Append mode: manifest + docs grow across resumes.
        with self.docs_path.open("a", encoding="utf-8") as docs_f, \
             self.manifest_path.open("a", encoding="utf-8") as man_f, \
             self.flagged_path.open("a", encoding="utf-8") as flag_f:
            for i in range(0, len(todo), self.batch_size):
                batch = todo[i:i + self.batch_size]
                fetched = self._fetch_revisions([t for _, t in batch])
                for _, title in batch:
                    if title not in fetched:
                        continue
                    page_id = fetched[title]["page_id"]
                    wikitext = fetched[title]["wikitext"]
                    text = strip_wikitext(wikitext, self._drop_prefixes)
                    char_count = len(text)
                    if char_count == 0:
                        stats.empty += 1
                    word_count = len(text.split())
                    if self.chars_per_token:
                        token_est = round(char_count / self.chars_per_token)
                    else:
                        token_est = round(word_count * self.fertility)

                    doc = {"page_id": page_id, "title": title, "text": text}
                    if keep_wikitext:
                        doc["wikitext"] = wikitext
                    docs_f.write(json.dumps(doc, ensure_ascii=False) + "\n")

                    record = {
                        "source_bucket": "manual",
                        "source_name": self.source_name,
                        "title": title,
                        "page_id": page_id,
                        "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                        "char_count": char_count,
                        "token_count_estimate": token_est,
                    }
                    # Flag suspiciously short docs for review instead of dropping.
                    if char_count < self.min_chars:
                        record["flagged_short"] = True
                        flag_f.write(json.dumps(
                            {"title": title, "page_id": page_id, "char_count": char_count},
                            ensure_ascii=False) + "\n")
                        stats.flagged_short += 1
                    # Flag any residual markup that survived all cleaning steps
                    # (see detect_residual_markup docstring) -- kept in the
                    # corpus, but visibly marked for T1.6 review, not silently
                    # trusted as clean.
                    leaks = detect_residual_markup(text, self._drop_prefixes)
                    if leaks:
                        record["flagged_residual_markup"] = leaks
                        flag_f.write(json.dumps(
                            {"title": title, "page_id": page_id, "residual_markup": leaks},
                            ensure_ascii=False) + "\n")
                        stats.flagged_residual_markup += 1
                    man_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats.fetched += 1
                # Flush each batch so a kill loses at most one batch of buffering.
                docs_f.flush(); man_f.flush(); flag_f.flush()
                print(f"  fetched {stats.fetched:,}/{len(todo):,} "
                      f"(flagged short: {stats.flagged_short})", file=sys.stderr)
        return stats


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    cfg = _load_tree_config()
    wiki_src = next((s for s in cfg.data.sources if s.name == "wiki"), None)
    if wiki_src is None or not wiki_src.wiki_lang:
        raise RuntimeError("config has no 'wiki' source with a wiki_lang.")

    ap = argparse.ArgumentParser(description=f"Wikipedia scraper for {cfg.lang_name} ({wiki_src.wiki_lang}).")
    ap.add_argument("--limit", type=int, default=None, help="max titles to fetch (testing).")
    ap.add_argument("--titles-only", action="store_true", help="only enumerate titles, then stop.")
    ap.add_argument("--keep-wikitext", action="store_true", help="also store raw wikitext in docs.jsonl.")
    ap.add_argument("--min-chars", type=int, default=200, help="flag-for-review threshold.")
    ap.add_argument("--min-interval", type=float, default=0.5, help="seconds between API requests.")
    args = ap.parse_args(argv)

    scraper = WikiScraper(
        wiki_lang=wiki_src.wiki_lang,
        raw_dir=Path(cfg.paths.raw_dir),
        manifest_path=Path(cfg.paths.manifest),
        source_name=f"wikipedia_{cfg.lang_code}",
        min_interval=args.min_interval,
        min_chars=args.min_chars,
        fertility=1.5 if cfg.lang_code == "hi" else 1.4,
        # Use the config's chars_per_token only when it's been explicitly set away
        # from the generic 4.0 default (e.g. Nepali's measured 4.988) -- keeps
        # Hindi's already-reported word-fertility numbers exactly as they were.
        chars_per_token=cfg.data.chars_per_token if cfg.data.chars_per_token != 4.0 else None,
    )
    scraper.enumerate_titles()
    if args.titles_only:
        return
    stats = scraper.fetch_all(limit=args.limit, keep_wikitext=args.keep_wikitext)
    print(f"done: fetched={stats.fetched} flagged_short={stats.flagged_short} "
          f"flagged_residual_markup={stats.flagged_residual_markup} empty={stats.empty}")


if __name__ == "__main__":
    main()
