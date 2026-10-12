import pandas as pd

KB = 8.617333262e-5            # eV K-1
HBAR = 1.054571817e-34         # J s
EV = 1.602176634e-19           # J
G = 6.241509074e-3             # eV per GPa A^3
T25 = 298.15                   # K

# Shannon radii (A)
RADII = {"Li": 0.76, "P": 0.17, "S": 1.84, "Se": 1.98, "O": 1.40, "Cl": 1.81, "Br": 1.96}
ANIONS = ("S", "Se", "O", "Cl", "Br")

FAMILIES = ("LGPS-type", "thio-LISICON")
EPS_INPUTS = {
    "LGPS-type": ["dX2_M", "dX2_P", "TdS_mix", "dr2_M", "dr2_P", "U_cation", "U_anion", "dH_mix"],
    "thio-LISICON": ["dX2_M", "TdS_mix", "dr2_M", "U_cation", "U_anion", "dH_mix"],
}
THIO_FACTOR = 25.0

# reference cell of the LGPS-type sigma-predictor: Li10GeP2S12, ICSD 188887
LGPS_REF = {"density_g_cm3": 2.0357647, "atoms": 50.056, "volume_A3": 960.82893, "formula_atoms": 25, "Z": 2}
POISSON = 0.25
DFT_B, DFT_G = 30.36, 14.35    # GPa

# LGSSPS (this work): Li10GexSiySnzP2S12
LGSSPS = pd.DataFrame({
    "composition": ["Li10Ge0.3Si0.3Sn0.4P2S12", "Li10Ge0.3Si0.15Sn0.55P2S12", "Li10Ge0.55Si0.2Sn0.25P2S12",
                    "Li10Ge0.8Si0.1Sn0.1P2S12", "Li10Ge0.3Si0.5Sn0.2P2S12"],
    "eps_exp": [0.0623, 0.0670, 0.0594, 0.0589, 0.0507],
    "sigma_pellet1": [6.970397, 5.917237, 6.205874, 5.405481, 4.938465],
    "sigma_pellet2": [7.134401, 5.805672, 6.345750, 5.135288, 4.403866],
    "density_pellet1": [1.642854257, 1.682752482, 1.631779943, 1.650322563, 1.621558483],
    "density_pellet2": [1.656865807, 1.662784546, 1.628274508, 1.671801815, 1.635674334],
    "density_theoretical": [1.994, 2.0285, 1.9945, 1.995, 1.948]})

# LGSSPSBrO: Li9.54[GexSi1-x-ySny]1.74P1.44S11.1Br0.3O0.6, sigma at 25 C (mS cm-1)
LGSSPSBRO = pd.DataFrame({
    "label": ["Si"] + [f"Ge x={0.1 * i:.1f}" for i in range(1, 11)] + [f"Sn y={0.1 * i:.1f}" for i in range(1, 7)],
    "sigma_exp": [10.9, 12.3, 11.3, 13.5, 13.6, 13.8, 12.7, 11.1, 7.9, 5.4, 4.1, 11.5, 8.8, 7.7, 6.7, 4.9, 4.5]})
