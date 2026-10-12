# ILFS

Code for reproducing the results of *Intrinsic Lattice Formation Strain Guides Screening of Sulfide Solid Electrolytes*.

## Setup

```
pip install -r requirements.txt
```

Place `Data S1.xlsx` and `Data S2.xlsx` (Supplementary Data) in `data/`.

## Usage

```
python reproduce.py                 # all tasks
python reproduce.py sigma table_s8  # selected tasks
```

Results are written to `results/`.

| Task | Results |
|---|---|
| `eps` | ε-predictor (Fig. S4), Table S3, ε_p ranges |
| `sigma` | Table 1, σ-predictor performance (Fig. S6) |
| `table_s2` | Table S2 |
| `table_s4` | Table S4 |
| `table_s5` | Table S5, single-sample calibration (Fig. 7h, Fig. S10) |
| `entropy` | LGSSPSBrO entropy and ordering (Fig. 7) |
| `table_s6` | Table S6, Fig. S11, Li-site correlations |
| `table_s7` | Table S7 (Fig. S7) |
| `table_s8` | Table S8 |
| `table_s9` | Table S9 |
| `pooled` | LOCO over both families |
| `calibration` | Single-sample calibration for every anchor (Fig. S9) |
