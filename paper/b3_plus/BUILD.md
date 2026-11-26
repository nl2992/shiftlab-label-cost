# Building the manuscript

First regenerate all results from the configured 20 datasets:

```bash
python paper/b3_plus/figures/make_all_streams.py
```

Then compile:

```bash
cd paper/b3_plus
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

TeX Live was installed here with `brew install texlive`, which needs no administrator rights
and puts `pdflatex` in `/opt/homebrew/bin`.

The paper uses the TMLR style (`tmlr.sty`, `tmlr.bst`, with its `fancyhdr.sty`), committed beside
the source along with `algorithm.sty` and `algorithmic.sty`.

`\usepackage{tmlr}` is the anonymous submission form. Use `[preprint]` or `[accepted]` to reveal the
author block.

## What the build does and does not check

Compiling confirms the document is well formed and tells you the page count, which is the one
thing static checking cannot establish. It does **not** check that the reported numbers are
correct. Those are in `tables/*.tex`; check them against the per-episode records in
`docs/promising_directions/results/b3_plus/`.

Build artifacts are gitignored, including `main.pdf`: the source and the generated tables are the
record, and a committed PDF goes stale the moment either changes.

## Writing and figure conventions

The prose revision follows [HKUSTDial paper-polish](https://github.com/HKUSTDial/Supervisor-Skills/tree/main/skills/paper-polish); result figures follow its [figure-designer](https://github.com/HKUSTDial/Supervisor-Skills/tree/main/skills/figure-designer) guidance. The plotting style follows [figures4papers](https://github.com/ChenLiu-1996/figures4papers/tree/main/scientific-figure-making): consistent blue/neutral/red palette, clean axes, black-edged bars, direct numeric labels, vector PDF assets and 300-dpi PNG previews. Mean and median use distinct line styles in dataset plots.

Regenerate the method figure separately with `python paper/b3_plus/figures/make_figure1.py`. The main results script regenerates all other result figures and the mean/median summary table. Headline savings are arithmetic means of dataset-level percentage reductions, with equal weight for all 20 datasets; medians use the same dataset-level reductions.
