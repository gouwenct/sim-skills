# NOTICE

This skill contains a Python adaptation of the ImDecorr decorrelation method:

- Descloux A, Grussmayer KS, Radenovic A. Parameter-free image resolution
  estimation based on decorrelation analysis. Nature Methods (2019).
- Historical source used for the adaptation:
  `ImDecorr-master_20250124/funcs` from the local archived MP-SIM analysis tree.
- Original code copyright (C) 2018 Adrien Descloux, EPFL LBEN.
- The adapted code is distributed under GNU GPL version 3 or later.

The Python baseline removes GUI execution, preserves float32 negative values,
uses deterministic montage placement, and rejects sector peaks that hit the
refined search boundary. It does not claim numerical identity with MATLAB.
