"""Hindi Wikisource scraper (T1.4b: supplementary manual tier).

A separate module from wiki_scrape.py on purpose, not just for organisation:
hi.wikisource.org's main-namespace pages are transclusion STUBS
(`<pages index="..." from=X to=Y />`) that pull the real text from the
Page: namespace at RENDER time. Fetching raw wikitext via
`action=query&prop=revisions` (wiki_scrape.py's approach) returns almost
nothing -- confirmed empirically: a real article's raw wikitext was 261
chars, nearly all of it the transclusion directive itself. The fix is
`action=parse&prop=text`, which asks the server to fully render the page
(resolving the transclusion) and returns HTML, not wikitext -- so this
module extracts text with BeautifulSoup (the news_scrape.py pattern), not
mwparserfromhell (the wiki_scrape.py pattern).

License: hi.wikisource.org is CC BY-SA 4.0 (confirmed via
action=query&meta=siteinfo&siprop=rightsinfo before building this), and
Wikisource specifically hosts public-domain/freely-licensed source texts --
about as clearly "permissively licensed" as a manual source gets.

Pipeline (mirrors wiki_scrape.py's resumability contract):
  1. enumerate_titles -> corpus/raw/wikisource_hi.titles.jsonl
  2. fetch_all        -> corpus/raw/wikisource_hi.docs.jsonl
                      -> corpus/raw/manifest.jsonl (source_bucket="manual")
                      -> corpus/raw/wikisource_hi.flagged.jsonl (over-short)

Resumability: "done" = page_ids already in the manifest for this source_name.
Etiquette: action=parse is a heavier server-side call than prop=revisions (it
renders the page), so this uses a slower default interval than wiki_scrape.py.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

TREE = Path(__file__).resolve().parent.parent
CONTACT = "anuragkaushal101@gmail.com"
API = "https://hi.wikisource.org/w/api.php"

# Header/footer/navigation chrome injected by the ProofreadPage extension
# around the real transcluded text -- confirmed by inspecting real rendered
# output before writing this (see module docstring).
_NAV_SELECTORS = ".ws-noexport, .noprint, #headerContainer, #footerContainer"
_MULTINL_RE = re.compile(r"\n{3,}")
# Residue sweep: found at full-scale verification (23/4,634 docs, ~0.5%) --
# a MediaWiki extension tag (`<section end="1" ... />`, used by the
# ProofreadPage LST extension) and a malformed source tag (`<poem>...<poem>`,
# missing the closing slash) both survive BeautifulSoup's HTML parse as
# literal text rather than being converted to real, removable elements.
# After soup.get_text(), anything still shaped like `<...>` is definitionally
# leaked markup (real tags are already gone), so this is safe to strip.
_TAG_RESIDUE_RE = re.compile(r"<[^<>\n]{0,200}>")
_LEAK_CHECK_RE = re.compile(r"<[^\s<>][^<>\n]{0,60}>")


def detect_residual_markup(text: str) -> bool:
    """True if a leak signature still remains after cleaning -- flag for
    review rather than trust silently, same philosophy as wiki_scrape.py's
    detect_residual_markup()."""
    return bool(_LEAK_CHECK_RE.search(text))


def _load_tree_config():
    """Import hindi/config.py and load hindi/configs/hindi.yaml (same pattern
    as wiki_scrape.py, duplicated here rather than imported -- see module
    docstring for why this file doesn't share code with wiki_scrape.py)."""
    import importlib.util
    cfg_py = TREE / "config.py"
    spec = importlib.util.spec_from_file_location(f"{TREE.name}_config", cfg_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    yamls = sorted((TREE / "configs").glob("*.yaml"))
    if len(yamls) != 1:
        raise RuntimeError(f"expected exactly one config yaml in {TREE/'configs'}, found {yamls}")
    return mod.load_config(yamls[0])


def extract_text(html: str) -> str:
    """Render-HTML -> clean plain text: strip ProofreadPage nav chrome, keep
    the transcluded body. Verified against real hi.wikisource.org pages."""
    soup = BeautifulSoup(html, "lxml")
    for el in soup.select(_NAV_SELECTORS):
        el.decompose()
    text = soup.get_text("\n", strip=True)
    text = _TAG_RESIDUE_RE.sub("", text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text = "\n".join(lines)
    return _MULTINL_RE.sub("\n\n", text).strip()


@dataclass
class ScrapeStats:
    fetched: int = 0
    skipped_done: int = 0
    flagged_short: int = 0
    flagged_residual_markup: int = 0
    empty: int = 0


class WikisourceScraper:
    """Enumerate + render + clean hi.wikisource.org, resumably."""

    def __init__(
        self,
        raw_dir: Path,
        manifest_path: Path,
        source_name: str = "wikisource_hi",
        *,
        min_interval: float = 1.0,   # slower than wiki_scrape.py: parse renders server-side
        max_retries: int = 5,
        min_chars: int = 100,        # Wikisource entries are often short by nature; flag not drop
        fertility: float = 1.5,
        timeout: int = 30,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.manifest_path = Path(manifest_path)
        self.source_name = source_name
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.min_chars = min_chars
        self.fertility = fertility
        self.timeout = timeout

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.titles_path = self.raw_dir / "wikisource_hi.titles.jsonl"
        self.docs_path = self.raw_dir / "wikisource_hi.docs.jsonl"
        self.flagged_path = self.raw_dir / "wikisource_hi.flagged.jsonl"

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                f"LMA-IndividualProject/0.1 (academic coursework; monolingual LM corpus; "
                f"contact {CONTACT}) python-requests/{requests.__version__}"
            )
        })
        self._last_request = 0.0

    def _get(self, params: dict) -> dict:
        params = {**params, "format": "json", "maxlag": 5}
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(API, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                self._sleep(min(2 ** attempt, 30), f"network error: {exc}")
                continue
            if resp.status_code == 429:
                self._sleep(float(resp.headers.get("Retry-After", 5)), "HTTP 429")
                continue
            if resp.status_code >= 500:
                self._sleep(min(2 ** attempt, 30), f"HTTP {resp.status_code}")
                continue
            data = resp.json()
            if "error" in data and data["error"].get("code") == "maxlag":
                self._sleep(5, "maxlag")
                continue
            return data
        raise RuntimeError(f"API request failed after {self.max_retries} retries: {params}")

    def _sleep(self, seconds: float, reason: str) -> None:
        print(f"  [wait {seconds:.0f}s] {reason}", file=sys.stderr)
        time.sleep(seconds)

    def enumerate_titles(self, force: bool = False) -> int:
        """List every main-namespace, non-redirect title (same list=allpages
        pattern as wiki_scrape.py)."""
        if self.titles_path.exists() and not force:
            n = sum(1 for _ in self.titles_path.open(encoding="utf-8"))
            print(f"titles.jsonl already present ({n:,}); skipping enumeration.")
            return n
        count = 0
        apcontinue = None
        with self.titles_path.open("w", encoding="utf-8") as out:
            while True:
                params = {"action": "query", "list": "allpages", "apnamespace": 0,
                          "apfilterredir": "nonredirects", "aplimit": 500}
                if apcontinue:
                    params["apcontinue"] = apcontinue
                data = self._get(params)
                for page in data["query"]["allpages"]:
                    out.write(json.dumps({"page_id": page["pageid"], "title": page["title"]},
                                         ensure_ascii=False) + "\n")
                    count += 1
                if "continue" in data:
                    apcontinue = data["continue"]["apcontinue"]
                    print(f"  enumerated {count:,} titles…", file=sys.stderr)
                else:
                    break
        print(f"enumerated {count:,} titles -> {self.titles_path.name}")
        return count

    def _done_page_ids(self) -> set[int]:
        done: set[int] = set()
        if self.manifest_path.exists():
            for line in self.manifest_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("source_name") == self.source_name and "page_id" in rec:
                    done.add(rec["page_id"])
        return done

    def fetch_all(self, limit: int | None = None) -> ScrapeStats:
        """Render + clean + record every enumerated title not already done.

        One page per action=parse call (the API does not batch parse
        requests the way prop=revisions batches). Stops early once `limit`
        NEW documents have been fetched, if given -- lets a run be capped to
        "collect enough to close the gap" rather than the full 6k titles.
        """
        if not self.titles_path.exists():
            raise RuntimeError("run enumerate_titles() first.")

        done = self._done_page_ids()
        print(f"resuming: {len(done):,} already in manifest.")

        todo: list[tuple[int, str]] = []
        for line in self.titles_path.open(encoding="utf-8"):
            rec = json.loads(line)
            if rec["page_id"] not in done:
                todo.append((rec["page_id"], rec["title"]))
        print(f"available to fetch: {len(todo):,} titles" + (f" (will stop after {limit:,} new docs)" if limit else ""))

        stats = ScrapeStats(skipped_done=len(done))
        with self.docs_path.open("a", encoding="utf-8") as docs_f, \
             self.manifest_path.open("a", encoding="utf-8") as man_f, \
             self.flagged_path.open("a", encoding="utf-8") as flag_f:
            for page_id, title in todo:
                if limit is not None and stats.fetched >= limit:
                    print(f"reached limit of {limit:,} new docs; stopping.")
                    break
                data = self._get({"action": "parse", "pageid": page_id, "prop": "text"})
                if "error" in data:
                    continue  # e.g. page deleted since enumeration; skip, don't crash
                html = data.get("parse", {}).get("text", {}).get("*", "")
                text = extract_text(html)
                char_count = len(text)
                if char_count == 0:
                    stats.empty += 1
                token_est = round(len(text.split()) * self.fertility)

                docs_f.write(json.dumps({"page_id": page_id, "title": title, "text": text},
                                        ensure_ascii=False) + "\n")
                record = {
                    "source_bucket": "manual", "source_name": self.source_name,
                    "title": title, "page_id": page_id,
                    "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                    "char_count": char_count, "token_count_estimate": token_est,
                }
                if char_count < self.min_chars:
                    record["flagged_short"] = True
                    flag_f.write(json.dumps({"title": title, "page_id": page_id, "char_count": char_count},
                                            ensure_ascii=False) + "\n")
                    stats.flagged_short += 1
                if detect_residual_markup(text):
                    record["flagged_residual_markup"] = True
                    flag_f.write(json.dumps({"title": title, "page_id": page_id, "reason": "residual_markup"},
                                            ensure_ascii=False) + "\n")
                    stats.flagged_residual_markup += 1
                man_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.fetched += 1
                if stats.fetched % 20 == 0:
                    docs_f.flush(); man_f.flush(); flag_f.flush()
                    print(f"  fetched {stats.fetched:,} (flagged short: {stats.flagged_short})", file=sys.stderr)
        return stats


def main(argv: list[str] | None = None) -> None:
    cfg = _load_tree_config()
    ap = argparse.ArgumentParser(description=f"Wikisource scraper for {cfg.lang_name}.")
    ap.add_argument("--limit", type=int, default=None, help="stop after this many NEW docs.")
    ap.add_argument("--titles-only", action="store_true")
    ap.add_argument("--min-interval", type=float, default=1.0)
    args = ap.parse_args(argv)

    scraper = WikisourceScraper(
        raw_dir=Path(cfg.paths.raw_dir), manifest_path=Path(cfg.paths.manifest),
        min_interval=args.min_interval,
    )
    scraper.enumerate_titles()
    if args.titles_only:
        return
    stats = scraper.fetch_all(limit=args.limit)
    print(f"done: fetched={stats.fetched} flagged_short={stats.flagged_short} "
          f"flagged_residual_markup={stats.flagged_residual_markup} empty={stats.empty}")


if __name__ == "__main__":
    main()
