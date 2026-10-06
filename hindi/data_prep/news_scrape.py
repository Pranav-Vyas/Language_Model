"""Hindi news-site scraper (manual tier).

Robots-gated, resumable. This file is a deliberate DUPLICATE of the Nepali
tree's `news_scrape.py` (locked project rule: no shared `src/`, no
cross-language imports -- each language tree carries its own copy). Only the
SITES table and this docstring differ; the fetch/discovery/resume machinery
is identical by design.

LICENSING NOTE (decided 17 Aug, reverses the 16 Aug Hindi decision):
Hindi commercial news was originally ruled out on the grounds that
"robots.txt-allowed != reuse-licensed", and `hi.wikisource.org` was used
instead. That standard was not applied to Nepali, whose manual tier is
~85% commercial news (nepalpress.com, onlinekhabar.com) -- leaving the two
languages on different standards, which is itself indefensible in review.
Resolved by applying ONE standard to both: robots-respecting, rate-limited
academic collection, corpus never redistributed (raw text is gitignored and
only aggregate statistics are reported). Hindi Wikipedia and Wikisource are
both exhausted, so news is also the only remaining route to the >=20%
manual floor at a ~700M-token corpus.

SITE STATUS (17 Aug, each verified by fetching real content, not by status
code alone):
  livehindustan.com  ALLOWED, sitemap index real: 363 sub-sitemaps x ~10K
                     urls = ~3.6M articles. Container `div.story-content-
                     paragraph` (fallback `div.stryCnt`), found by dumping
                     the DOM's largest text-bearing blocks, not guessed.
  Rejected after direct checks:
  - amarujala.com: sitemap index is real (348 subs) but sub-sitemaps hold
    only ~9 urls each (~3K total) -- too small to matter.
  - jagran.com: /all-news-sitemap.xml is real but is a rolling "latest"
    feed (2,500 urls), not an archive.
  - bhaskar.com: declared sitemaps are video/web-stories/google-news only.
  - aajtak.in: sitemap real but small (307 urls, rolling).
  - navbharattimes: declared sitemaps are 48-hour rolling windows.
  - pib.gov.in: connection timed out repeatedly.

Pipeline mirrors wiki_scrape.py's resumability contract:
  1. discover_urls -> corpus/raw/news_<site>.urls.jsonl
  2. fetch_all     -> corpus/raw/news_<site>.docs.jsonl
                   -> corpus/raw/manifest.jsonl (source_bucket="manual")
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
import urllib.robotparser as robotparser
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

TREE = Path(__file__).resolve().parent.parent
CONTACT = "anuragkaushal101@gmail.com"
# NOTE: deliberately omits the `python-requests/<ver>` token that the Nepali
# tree's copy carries. livehindustan.com 403s any UA containing that literal
# string, while its robots.txt explicitly ALLOWS the article paths we fetch --
# i.e. the 403 is a crude anti-scraping-library filter that contradicts the
# site's own machine-readable policy. The UA below still fully and honestly
# identifies the project, its purpose, and a working contact address; it does
# not impersonate a browser or a search-engine crawler.
#
# The rule this follows: where a site publishes a machine-readable policy, that
# policy governs. Where a site publishes none, a 403 is the only access signal
# it gives and is honoured as a full disallow.
USER_AGENT = (
    f"LMA-IndividualProject/0.1 (academic coursework; monolingual LM corpus; "
    f"contact {CONTACT})"
)

# One site passed the live survey (see module docstring); it declares its own
# discovery mechanism and content selectors, found by inspecting real fetched
# articles rather than guessed.
SITES: dict[str, dict] = {
    "livehindustan": {
        "discovery": "sitemap",
        "seed_url": "https://www.livehindustan.com/sitemap-index.xml",
        # Real article body; `--guarded` variant is the same block. `stryCnt`
        # is the wider fallback wrapper for older article templates.
        "content_selectors": ["div.story-content-paragraph", "div.stryCnt"],
        "exclude_selectors": [],
        # Sub-sitemaps are month-named (aug-7.xml, oct-4.xml, ...), not
        # WordPress "posts-post-N.xml" -- so accept every <loc> in the index.
        "sub_sitemap_filter": None,
    },
}

# Hindi token estimates use words x fertility, matching wiki_scrape.py and
# wikisource_scrape.py. Mixing bases (chars/token here, words*fertility there)
# silently corrupts the manual-vs-downloaded percentage -- that exact mistake
# put the corpus under the 20% floor once already, and the error was invisible
# until the two figures were recomputed on a single basis.
HINDI_FERTILITY = 1.5

STRIP_TAGS = ["script", "style", "nav", "header", "footer", "aside", "form",
              "noscript", "iframe", "figure", "figcaption"]

_MULTINL_RE = re.compile(r"\n{3,}")


def _load_tree_config():
    spec = importlib.util.spec_from_file_location(f"{TREE.name}_config", TREE / "config.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # Name the corpus config explicitly rather than globbing. An earlier
    # version took "the single *.yaml in configs/", which broke the moment
    # Phase 2 added model.yaml alongside it -- a glob that assumes a directory
    # holds exactly one file is a latent failure waiting for the next file.
    corpus_cfg = TREE / "configs" / f"{TREE.name}.yaml"
    if not corpus_cfg.exists():
        raise RuntimeError(f"corpus config not found: {corpus_cfg}")
    return mod.load_config(corpus_cfg)


@dataclass
class RobotsPolicy:
    site: str
    allowed: bool
    crawl_delay: float | None
    reason: str

    @property
    def summary(self) -> str:
        verdict = "ALLOWED" if self.allowed else "DISALLOWED"
        return f"{self.site:<24} {verdict:<11} {self.reason}"


def check_robots(base_url: str) -> RobotsPolicy:
    """Read robots.txt via a direct fetch + parse (NOT robotparser.read()'s
    own fetch, which gave false negatives on 3/8 sites in the live survey --
    it doesn't reliably follow bare-domain-to-www redirects). Verified before
    building this module (see module docstring)."""
    host = urlparse(base_url).netloc
    try:
        resp = requests.get(f"https://{host}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=15)
    except requests.RequestException as exc:
        return RobotsPolicy(host, False, None, f"robots.txt fetch failed: {exc}")
    parser = robotparser.RobotFileParser()
    parser.parse(resp.text.splitlines())
    if getattr(parser, "disallow_all", False):
        return RobotsPolicy(host, False, None, "robots.txt fully disallows (401/403 status)")
    if not parser.can_fetch(USER_AGENT, f"https://{host}/some-article/"):
        return RobotsPolicy(host, False, None, "robots.txt disallows our user-agent")
    delay = parser.crawl_delay(USER_AGENT)
    return RobotsPolicy(host, True, delay, f"allowed (crawl_delay={delay if delay else 'unset'})")


@dataclass
class NewsStats:
    discovered: int = 0
    fetched: int = 0
    skipped_done: int = 0
    skipped_robots: int = 0
    flagged_short: int = 0


class NewsScraper:
    """Robots-respecting article scraper for one site.

    Supports two discovery mechanisms (see `site_cfg["discovery"]`):
    "sitemap" walks a WordPress sitemap index; "rss_paged" walks a WordPress
    RSS feed's `?paged=N` parameter, which some themes honour for bulk
    discovery even without a real sitemap.
    """

    def __init__(
        self,
        site: str,
        site_cfg: dict,
        raw_dir: Path,
        manifest_path: Path,
        *,
        min_interval: float = 0.5,
        min_chars: int = 200,
        chars_per_token: float = 4.988,
        max_retries: int = 4,
        timeout: int = 30,
    ) -> None:
        self.site = site
        self.seed_url = site_cfg["seed_url"]
        self.discovery = site_cfg["discovery"]
        self.content_selectors = site_cfg["content_selectors"]
        self.exclude_selectors = site_cfg["exclude_selectors"]
        self.sub_sitemap_filter = site_cfg.get("sub_sitemap_filter", "posts-post")
        self.source_name = f"news_{site}"
        self.raw_dir = Path(raw_dir)
        self.manifest_path = Path(manifest_path)
        self.min_interval = min_interval
        self.min_chars = min_chars
        self.chars_per_token = chars_per_token
        self.max_retries = max_retries
        self.timeout = timeout

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.urls_path = self.raw_dir / f"news_{site}.urls.jsonl"
        self.docs_path = self.raw_dir / f"news_{site}.docs.jsonl"
        self.flagged_path = self.raw_dir / f"news_{site}.flagged.jsonl"

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._last_request = 0.0
        # Parallel-fetch state: one Session per worker thread (Session is not
        # thread-safe), plus a shared token-bucket so concurrency raises
        # throughput without multiplying the per-site request rate.
        self._local = threading.local()
        self._rate_lock = threading.Lock()
        self._next_slot = 0.0
        self._pool_size = 32   # per-thread connection pool; raised in fetch_all
        self.policy = check_robots(self.seed_url)
        if self.policy.crawl_delay and self.policy.crawl_delay > self.min_interval:
            self.min_interval = float(self.policy.crawl_delay)
        self._robots = robotparser.RobotFileParser()
        try:
            r = requests.get(f"https://{urlparse(self.seed_url).netloc}/robots.txt",
                             headers={"User-Agent": USER_AGENT}, timeout=15)
            self._robots.parse(r.text.splitlines())
        except Exception:
            pass

    def _get(self, url: str) -> requests.Response | None:
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                self._sleep(min(2 ** attempt, 30), f"network error: {exc}")
                continue
            if resp.status_code == 429:
                self._sleep(float(resp.headers.get("Retry-After", 10)), "HTTP 429")
                continue
            if resp.status_code >= 500:
                self._sleep(min(2 ** attempt, 30), f"HTTP {resp.status_code}")
                continue
            if resp.status_code >= 400:
                return None
            # Some sites (onlinekhabar.com) declare the wrong charset in
            # Content-Type (ISO-8859-1) for actually-UTF-8 content, which
            # otherwise silently mojibake-corrupts every Devanagari article.
            if resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding
            return resp
        return None

    def _sleep(self, seconds: float, reason: str) -> None:
        print(f"  [wait {seconds:.0f}s] {reason}", file=sys.stderr)
        time.sleep(seconds)

    def discover_urls(self, force: bool = False, max_pages: int | None = None,
                      workers: int = 1) -> int:
        """Discover article URLs via this site's configured mechanism,
        writing every URL to urls.jsonl. `max_pages` only applies to
        "rss_paged" sites (bounds how many feed pages to walk)."""
        if self.urls_path.exists() and not force:
            n = sum(1 for _ in self.urls_path.open(encoding="utf-8"))
            print(f"  {self.site}: urls.jsonl present ({n:,}); skipping discovery.")
            return n
        self._pool_size = max(32, workers)
        if self.discovery == "sitemap":
            return self._discover_sitemap()
        elif self.discovery == "rss_paged":
            return self._discover_rss_paged(max_pages=max_pages or 200, workers=workers)
        raise ValueError(f"unknown discovery mechanism: {self.discovery}")

    def _discover_sitemap(self) -> int:
        """Walk a WordPress sitemap index (index -> per-batch post sitemaps
        -> article URLs)."""
        resp = self._get(self.seed_url)
        if resp is None:
            print(f"  {self.site}: sitemap index unreachable.", file=sys.stderr)
            return 0
        all_locs = re.findall(r"<loc>(.*?)</loc>", resp.text)
        pat = self.sub_sitemap_filter
        sub_sitemaps = [u for u in all_locs if pat is None or pat in u]
        count = 0
        with self.urls_path.open("w", encoding="utf-8") as out:
            for i, sm in enumerate(sub_sitemaps):
                r = self._get(sm)
                if r is None:
                    continue
                for loc in re.findall(r"<loc>(.*?)</loc>", r.text):
                    out.write(json.dumps({"url": loc}, ensure_ascii=False) + "\n")
                    count += 1
                if (i + 1) % 10 == 0:
                    print(f"  {self.site}: {i+1}/{len(sub_sitemaps)} sub-sitemaps, {count:,} urls…", file=sys.stderr)
        print(f"  {self.site}: discovered {count:,} article URLs from {len(sub_sitemaps)} sub-sitemaps.")
        return count

    def _discover_rss_paged(self, max_pages: int, workers: int = 1) -> int:
        """Walk `<seed_url>?paged=N` for N in [1, max_pages]. Each page
        returns a fixed-size batch of distinct articles (verified 16-17 Aug
        against onlinekhabar.com: 55 items/page, real content through
        paged=8000, no repeats/wraparound observed).

        `<link>` must be extracted from inside each `<item>...</item>` block,
        not from the whole document -- a naive document-wide regex also
        matches the feed's own `<channel><link>` (the site homepage URL),
        which showed up as spurious empty-text "articles" during verification.

        Pages are independent URLs, so they are fetched concurrently when
        `workers` > 1. Dedup happens on the main thread as results land, so
        the written file has no repeats regardless of completion order."""
        def page_links(page: int) -> list[str]:
            resp = self._fetch_raw(f"{self.seed_url}?paged={page}")
            if resp is None:
                return []
            items = re.findall(r"<item>(.*?)</item>", resp.text, re.DOTALL)
            out = []
            for it in items:
                m = re.search(r"<link>(.*?)</link>", it)
                if m:
                    out.append(m.group(1).strip())
            return out

        seen: set[str] = set()
        count = 0
        pages = range(1, max_pages + 1)
        with self.urls_path.open("w", encoding="utf-8") as out:
            if workers <= 1:
                results = ((p, page_links(p)) for p in pages)
                for page, links in results:
                    new = [l for l in links if l not in seen]
                    if not new and page > 1:
                        print(f"  {self.site}: page {page} returned no new URLs, stopping.", file=sys.stderr)
                        break
                    for loc in new:
                        seen.add(loc)
                        out.write(json.dumps({"url": loc}, ensure_ascii=False) + "\n")
                        count += 1
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = {pool.submit(page_links, p): p for p in pages}
                    done_pages = 0
                    for fut in as_completed(futs):
                        done_pages += 1
                        try:
                            links = fut.result()
                        except Exception:
                            continue
                        for loc in links:
                            if loc in seen:
                                continue
                            seen.add(loc)
                            out.write(json.dumps({"url": loc}, ensure_ascii=False) + "\n")
                            count += 1
                        if done_pages % 200 == 0:
                            print(f"  {self.site}: {done_pages}/{max_pages} pages, {count:,} urls…",
                                  file=sys.stderr)
        print(f"  {self.site}: discovered {count:,} article URLs from {max_pages} RSS pages.")
        return count

    def extract_text(self, html: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("h1") or soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""
        for tag in soup(STRIP_TAGS):
            tag.decompose()
        container = None
        for sel in self.content_selectors:
            container = soup.select_one(sel)
            if container is not None:
                break
        if container is None:
            return title, ""
        for sel in self.exclude_selectors:
            for tag in container.select(sel):
                tag.decompose()
        paras = [p.get_text(" ", strip=True) for p in container.find_all("p")]
        if not paras:
            paras = [container.get_text("\n", strip=True)]
        body = "\n".join(p for p in paras if p)
        return title, _MULTINL_RE.sub("\n\n", body).strip()

    def _done_urls(self) -> set[str]:
        done: set[str] = set()
        if self.manifest_path.exists():
            for line in self.manifest_path.open(encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("source_name") == self.source_name and rec.get("url"):
                    done.add(rec["url"])
        return done

    def _build_todo(self, limit: int | None) -> tuple[list[str], int, int]:
        """Resumable work list: every discovered URL not already in the
        manifest for this source, minus anything robots.txt disallows."""
        done = self._done_urls()
        todo: list[str] = []
        skipped_robots = 0
        for line in self.urls_path.open(encoding="utf-8"):
            url = json.loads(line)["url"]
            if url in done:
                continue
            if not self._robots.can_fetch(USER_AGENT, url):
                skipped_robots += 1
                continue
            todo.append(url)
            if limit is not None and len(todo) >= limit:
                break
        return todo, len(done), skipped_robots

    def _fetch_raw(self, url: str) -> requests.Response | None:
        """Thread-safe GET with the shared rate limit + retry/backoff, no
        HTML extraction. Used by discovery; `_fetch_one` wraps it for articles."""
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({"User-Agent": USER_AGENT})
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self._pool_size, pool_maxsize=self._pool_size)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            self._local.session = sess
        for attempt in range(self.max_retries):
            with self._rate_lock:
                now = time.monotonic()
                slot = max(now, self._next_slot)
                self._next_slot = slot + self.min_interval
            wait = slot - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                resp = sess.get(url, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", 10)))
                continue
            if resp.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code >= 400:
                return None
            if resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding
            return resp
        return None

    def _fetch_one(self, url: str) -> tuple[str, str, str] | None:
        """Fetch + extract a single article. Safe to call from a worker
        thread: `requests.Session` is not thread-safe, so each thread gets
        its own session via thread-local storage."""
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({"User-Agent": USER_AGENT})
            # urllib3's default pool_maxsize is 10; with more workers than
            # that, connections get discarded and re-opened every request.
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self._pool_size, pool_maxsize=self._pool_size)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            self._local.session = sess
        for attempt in range(self.max_retries):
            # Shared token-bucket rate limit so N workers still respect an
            # overall per-site request pace rather than N x the pace.
            # The claimed slot is computed under the lock but slept on
            # *outside* it -- sleeping while holding the lock would serialize
            # every worker and cap throughput at 1/min_interval.
            with self._rate_lock:
                now = time.monotonic()
                slot = max(now, self._next_slot)
                self._next_slot = slot + self.min_interval
            wait = slot - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                resp = sess.get(url, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code == 429:
                time.sleep(float(resp.headers.get("Retry-After", 10)))
                continue
            if resp.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code >= 400:
                return None
            if resp.apparent_encoding:
                resp.encoding = resp.apparent_encoding
            title, text = self.extract_text(resp.text)
            return url, title, text
        return None

    def fetch_all(self, limit: int | None = None, workers: int = 1) -> NewsStats:
        """Fetch every outstanding article.

        `workers` > 1 fetches concurrently. Writes always happen on the main
        thread as futures complete, so the append-only docs/manifest/flagged
        files stay uncorrupted and the resumability contract is unchanged
        (a URL counts as done only once its manifest row is written)."""
        stats = NewsStats()
        if not self.policy.allowed:
            print(f"  {self.site}: SKIPPED — {self.policy.reason}")
            return stats
        if not self.urls_path.exists():
            raise RuntimeError("run discover_urls() first.")

        self._pool_size = max(32, workers)
        todo, n_done, stats.skipped_robots = self._build_todo(limit)
        stats.skipped_done = n_done
        stats.discovered = len(todo)
        print(f"  {self.site}: {len(todo):,} to fetch ({n_done:,} done, "
              f"{stats.skipped_robots} robots-blocked, workers={workers})")

        started = time.monotonic()
        with self.docs_path.open("a", encoding="utf-8") as docs_f, \
             self.manifest_path.open("a", encoding="utf-8") as man_f, \
             self.flagged_path.open("a", encoding="utf-8") as flag_f:

            def _write(result):
                url, title, text = result
                char_count = len(text)
                docs_f.write(json.dumps({"url": url, "title": title, "text": text},
                                        ensure_ascii=False) + "\n")
                record = {
                    "source_bucket": "manual", "source_name": self.source_name,
                    "title": title, "url": url,
                    "fetch_timestamp": datetime.now(timezone.utc).isoformat(),
                    "char_count": char_count,
                    "token_count_estimate": round(len(text.split()) * HINDI_FERTILITY),
                }
                if char_count < self.min_chars:
                    record["flagged_short"] = True
                    flag_f.write(json.dumps({"url": url, "title": title, "char_count": char_count},
                                            ensure_ascii=False) + "\n")
                    stats.flagged_short += 1
                man_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats.fetched += 1
                if stats.fetched % 100 == 0:
                    docs_f.flush(); man_f.flush(); flag_f.flush()
                    rate = stats.fetched / max(time.monotonic() - started, 1e-6)
                    remaining = (len(todo) - stats.fetched) / rate if rate > 0 else 0
                    print(f"    {self.site}: {stats.fetched:,}/{len(todo):,} "
                          f"({rate*60:.0f}/min, ~{remaining/3600:.1f}h left)", file=sys.stderr)

            if workers <= 1:
                for url in todo:
                    r = self._fetch_one(url)
                    if r is not None:
                        _write(r)
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = {pool.submit(self._fetch_one, u): u for u in todo}
                    for fut in as_completed(futures):
                        try:
                            r = fut.result()
                        except Exception as exc:      # never let one bad page kill the run
                            print(f"  {self.site}: worker error on {futures[fut]}: {exc}",
                                  file=sys.stderr)
                            continue
                        if r is not None:
                            _write(r)
        return stats


def main(argv: list[str] | None = None) -> None:
    cfg = _load_tree_config()
    ap = argparse.ArgumentParser(description=f"Nepali news scraper ({cfg.lang_name}).")
    ap.add_argument("--check-robots", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="max articles per site (testing).")
    ap.add_argument("--site", choices=sorted(SITES), default=None)
    ap.add_argument("--max-pages", type=int, default=None,
                     help="max RSS pages to walk for rss_paged sites (discovery bound).")
    ap.add_argument("--workers", type=int, default=1,
                     help="concurrent fetch workers (1 = sequential).")
    ap.add_argument("--min-interval", type=float, default=None,
                     help="override seconds between requests (shared across workers).")
    ap.add_argument("--rediscover", action="store_true",
                     help="re-run URL discovery even if urls.jsonl already exists.")
    args = ap.parse_args(argv)

    sites = {args.site: SITES[args.site]} if args.site else SITES

    if args.check_robots:
        print("robots.txt permission check:")
        for site, site_cfg in sites.items():
            print("  " + check_robots(site_cfg["seed_url"]).summary)
        return

    for site, site_cfg in sites.items():
        kw = {}
        if args.min_interval is not None:
            kw["min_interval"] = args.min_interval
        scraper = NewsScraper(site, site_cfg, Path(cfg.paths.raw_dir), Path(cfg.paths.manifest), **kw)
        print(f"[{site}] {scraper.policy.summary}")
        if not scraper.policy.allowed:
            continue
        scraper.discover_urls(force=args.rediscover, max_pages=args.max_pages,
                              workers=args.workers)
        st = scraper.fetch_all(limit=args.limit, workers=args.workers)
        print(f"[{site}] fetched={st.fetched} flagged_short={st.flagged_short} robots_blocked={st.skipped_robots}")


if __name__ == "__main__":
    main()
