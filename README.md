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
python reproduce.py sigma table_s9  # selected tasks
```

Results are written to `results/`.

| Task | Results |
|---|---|
| `eps` | ε-predictor (Fig. S5), Table S4, ε_p ranges |
| `vegard` | Comparison with Vegard's law |
| `sigma` | Table 1, σ-predictor performance (Fig. S7) |
| `table_s2` | Table S2, Fig. S3, Li-site correlations |
| `table_s3` | Table S3 |
| `table_s5` | Table S5 |
| `table_s6` | Table S6, single-sample calibration (Fig. 7h, Fig. S11) |
| `entropy` | LGSSPSBrO entropy and ordering (Fig. 7) |
| `table_s7` | Table S7 (Fig. S8) |
| `table_s8` | Table S8 |
| `table_s9` | Table S9 |
| `pooled` | LOCO over both families |
| `calibration` | Single-sample calibration for every anchor (Fig. S10) |
