# Pair 24, Eight Bits: Abalone agent (I2AI 26/27-F midterm)

Salih Eren, Eren Terakye. Department of Computer Engineering, Eskisehir Osmangazi University.
Repository with the same files: https://github.com/SalihEREN0/I2AI_final

- `Pair24_EightBits_Midterm.pdf`: the paper (IEEE conference format, 5 pages).
- `Pair24_EightBits_Supplementary.pdf`: supplementary material: complete results of Assignments 1-4, the
  algorithms, the faster engine and its verification, and where every number comes from.
- `paper/`: LaTeX sources of both PDFs and the scripts that make their figures and tables. Build:
  `tectonic Pair24_EightBits_Midterm.tex` (or pdfLaTeX with BibTeX), and the same for
  `ek-materyal/Pair24_EightBits_Supplementary.tex`. Figures: `python figures.py` (needs matplotlib and the Times New
  Roman font); supplementary tables: `python ek-materyal/tablolar.py` (standard library). Both read the results in
  `project/`.
- `project/`: the agent and all experiments (`project/README.md`). Python 3 standard library only (tested
  with Python 3.14):
  `cd project`, then `python -m unittest discover -s tests` (tests) and
  `PYTHONPATH=src python -m abalone.server` (the demo at http://localhost:8024; in PowerShell:
  `$env:PYTHONPATH="src"; python -m abalone.server`). Where the project's notes say "repository root", they mean
  the `project/` folder.
- `project/demo-video/`: a silent 4:09 screen recording of the developer view (in the repository and the larger zip
  only).
