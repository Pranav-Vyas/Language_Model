# Hindi — corpus collection summary

> **Collection-time log — not the final corpus accounting.**
>
> This file was written on 17 Aug 2026 *while collection was still running*, to
> size the remaining downloads. It therefore lists only the sources fetched by
> that moment, and its token counts are pre-tokenizer estimates (see the basis
> line below), not counts from the trained tokenizer.
>
> It is kept as a dated record of how the corpus was assembled. For the final
> figures use `hindi/corpus/stats/corpus_stats.json`, summarised in
> `report/phase1/README.md` §2–§3:
>
> | | this log (17 Aug) | final |
> |---|---:|---:|
> | documents | 760,347 | 1,202,072 |
> | tokens | 412,803,506 (est.) | 582,711,232 (tokenizer) |
> | manual share | 24.03% | 19.29% |
>
> The manual-share line below reflects only the sources present at the time of
> writing, so it is not the corpus's manual share. The final value is 19.29%.

Generated: 2026-08-17T08:39:21.499330+00:00

Token-estimate basis: `words x fertility` (fertility=1.5)

> These are PRE-tokenizer estimates used for collection sizing and the
> manual-vs-downloaded split. Final reported token counts come from the
> trained SentencePiece tokenizer.

## Per source

| source | bucket | docs | chars | est. tokens | flagged short | flagged markup | first fetch | last fetch |
|---|---|---:|---:|---:|---:|---:|---|---|
| `news_livehindustan` | manual | 56,734 | 33,033,154 | 8,759,531 | 2,539 | 0 | 2026-08-17T08:02:22 | 2026-08-17T08:39:18 |
| `sangraha_verified_hin` | downloaded | 524,288 | 1,149,715,983 | 313,601,111 | 0 | 0 | 2026-08-16T04:54:32 | 2026-08-16T04:56:09 |
| `wikipedia_hi` | manual | 174,691 | 274,288,723 | 75,058,421 | 46,658 | 92 | 2026-08-15T19:29:49 | 2026-08-15T22:59:21 |
| `wikisource_hi` | manual | 4,634 | 51,343,955 | 15,384,443 | 15 | 1 | 2026-08-16T05:32:37 | 2026-08-16T06:57:08 |

## Per bucket

| bucket | docs | est. tokens |
|---|---:|---:|
| downloaded | 524,288 | 313,601,111 |
| manual | 236,059 | 99,202,395 |

**Total documents:** 760,347

**Total estimated tokens:** 412,803,506

**Manual fraction:** 24.03% (spec floor: 20%) — PASS
