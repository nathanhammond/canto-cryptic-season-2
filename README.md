# Canto Cryptic season 2

`days/` contains all 32 pages rendered at 300 DPI from
`canto.cryptic season 2 standard.pdf`. The pages keep the orientation of the
source PDF. Day 16 has no cipher content in the PDF.

`characters/` contains the 23 normalized components supplied from day 31 rows
1-3 plus the subsequently confirmed missing component. They keep their source
orientation and need only one size. Each file uses a unique, lowercase,
similar-looking Latin letter:

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

The confirmed crossed/arched component is `f.png`. The short horizontal
`l.png` component is accepted only at the start of a line or after a visible
whitespace run, preventing it from matching arbitrary pieces of a baseline.

Run the original template matcher with:

```sh
uv run --with-requirements requirements.txt process.py
```

The generated annotated pages are written to `output/`. The matcher detects the
horizontal cipher baselines below the Chinese/English heading, then greedily
matches each line from left to right. Colored rectangles therefore follow the
reading sequence without overlapping. Red rectangles mark currently unmatched
regions that need review.

`missing_symbols/` is rebuilt on each run with deduplicated review candidates.
`index.csv` records each candidate's detected occurrences and source
coordinates. Candidate images include 24 pixels of neighboring context on both
sides because adjacent components can share strokes. To avoid creating hundreds
of one-off segmentation files, the folder keeps the 32 most recurrent residual
shape groups. Day 16 is skipped because its PDF page contains no cipher content.
