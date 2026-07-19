# Canto Cryptic season 2

`days/` contains all 32 pages rendered at 300 DPI from
`canto.cryptic season 2 standard.pdf`. The pages keep the orientation of the
source PDF. Day 16 has no cipher content in the PDF.

`characters/` contains the 23 normalized components supplied from day 31 rows
1-3. They also keep their source orientation and need only one size. Each file
uses a unique, lowercase, similar-looking Latin letter:

| component | file | component | file |
| ---: | --- | ---: | --- |
| 0 | `m.png` | 12 | `d.png` |
| 1 | `w.png` | 13 | `q.png` |
| 2 | `v.png` | 14 | `e.png` |
| 3 | `n.png` | 15 | `c.png` |
| 4 | `k.png` | 16 | `p.png` |
| 5 | `a.png` | 17 | `x.png` |
| 6 | `g.png` | 18 | `s.png` |
| 7 | `b.png` | 19 | `y.png` |
| 8 | `h.png` | 20 | `i.png` |
| 9 | `o.png` | 21 | `t.png` |
| 10 | `r.png` | 22 | `l.png` |
| 11 | `j.png` |  |  |

Run the original template matcher with:

```sh
uv run --with-requirements requirements.txt process.py
```

The generated annotated pages are written to `output/`. The matcher detects the
horizontal cipher baselines below the Chinese/English heading, then greedily
matches each line from left to right. Colored rectangles therefore follow the
reading sequence without overlapping. Red rectangles mark occurrences of the
user-confirmed missing symbol.

`missing_symbols/candidate_01.png` is the confirmed missing symbol, and
`index.csv` records each detected occurrence and source coordinate. Day 16 is
skipped because its PDF page contains no cipher content.
