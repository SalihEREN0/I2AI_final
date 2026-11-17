# Pair 24, Eight Bits: Abalone agent (I2AI 26/27-F midterm)

Salih Eren, Eren Terakye. Department of Computer Engineering, Eskisehir Osmangazi University.
Repository with the same files: https://github.com/SalihEREN0/I2AI_final

- `Pair24_EightBits_Midterm.pdf`: the paper (IEEE conference format, 5 pages).
- `Pair24_EightBits_Supplementary.pdf`: supplementary material: complete results of Assignments 1-4, the
  algorithms, the faster engine and its verification, and where every number comes from.
- `paper/`: LaTeX sources of both PDFs and the scripts that make their figures and tables.
- `project/`: the agent and all experiments (`project/README.md`). Python 3 standard library only:
  `cd project`, then `python -m unittest discover -s tests` (tests) and
  `PYTHONPATH=src python -m abalone.server` (the demo at http://localhost:8024).
- `project/demo-video/`: a 4:09 recording of the developer view (only in the zip with the video).
