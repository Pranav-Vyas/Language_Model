# Nepali — corpus collection summary

> **Collection-time log — not the final corpus accounting.**
>
> This file was written on 17 Aug 2026 *while collection was still running*, to
> size the remaining downloads. It therefore lists only the sources fetched by
> that moment, and its token counts are pre-tokenizer estimates (see the basis
> line below), not counts from the trained tokenizer.
>
> It is kept as a dated record of how the corpus was assembled. For the final
> figures use `nepali/corpus/stats/corpus_stats.json`, summarised in
> `report/phase1/README.md` §2–§3:
>
> | | this log (17 Aug) | final |
> |---|---:|---:|
> | documents | 136,047 | 2,180,424 |
> | tokens | 50,077,299 (est.) | 903,937,802 (tokenizer) |
> | manual share | 100.00% | 22.40% |
>
> The manual-share line below reflects only the sources present at the time of
> writing, so it is not the corpus's manual share. The final value is 22.40%.

Generated: 2026-08-17T08:39:22.215507+00:00

Token-estimate basis: `chars / chars_per_token` (chars_per_token=4.988)

> These are PRE-tokenizer estimates used for collection sizing and the
> manual-vs-downloaded split. Final reported token counts come from the
> trained SentencePiece tokenizer.

## Per source

| source | bucket | docs | chars | est. tokens | flagged short | flagged markup | first fetch | last fetch |
|---|---|---:|---:|---:|---:|---:|---|---|
| `news_nepalpress` | manual | 46,707 | 72,618,944 | 14,558,757 | 660 | 0 | 2026-08-16T10:13:37 | 2026-08-17T08:39:15 |
| `news_onlinekhabar` | manual | 58,924 | 132,923,342 | 26,648,487 | 70 | 0 | 2026-08-17T03:45:20 | 2026-08-17T08:39:21 |
| `wikipedia_ne` | manual | 30,416 | 44,244,138 | 8,870,055 | 4,629 | 24 | 2026-08-16T09:57:39 | 2026-08-17T04:33:08 |

## Per bucket

| bucket | docs | est. tokens |
|---|---:|---:|
| manual | 136,047 | 50,077,299 |

**Total documents:** 136,047

**Total estimated tokens:** 50,077,299

**Manual fraction:** 100.00% (spec floor: 20%) — PASS
