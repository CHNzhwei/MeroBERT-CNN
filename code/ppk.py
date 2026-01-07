import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def parse_frequency(freq_raw):
    if pd.isna(freq_raw):
        return None

    freq = str(freq_raw).strip().lower()

    if "q" in freq and "h" in freq:
        num = freq.replace("q", "").replace("h", "").replace(" ", "")
        try: return float(num)
        except: pass

    if freq.endswith("h"):
        num = freq.replace("h", "").strip()
        try: return float(num)
        except: pass

    if freq in ["qd", "q.d", "daily", "q24h"]:
        return 24.0

    try:
        return float(freq)
    except:
        return None
    
def preprocess_dataframe(df):
    df_processed = df.copy()
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean_val = df_processed[col].mean()
        df_processed[col] = df_processed[col].fillna(mean_val)
    return df_processed

# CKD-EPI 2021
def ckd_epi_2021(age, sex, scr_mgdl):
    female = (sex == 2)
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    scr_k = scr_mgdl / kappa

    return (
        142 *
        (min(scr_k,1)**alpha) *
        (max(scr_k,1)**(-1.2)) *
        (0.9938**age) *
        (1.012 if female else 1)
    )


# Lee 2021 PPK parameters
THETA1_CL = 6.37
THETA2_GFR = 0.00925
VC = 9.07
VP = 7.91
Q = 10.7
GFR_MEAN = 91.57

def simulate_trough(dose_mg, tinf_h, tau_h, CL, cycles=5):
    dt = 0.01
    total_time = tau_h * cycles
    steps = int(total_time / dt)

    A1 = 0.0
    A2 = 0.0

    def infusion(t):
        m = t % tau_h
        return dose_mg / tinf_h if 0 <= m <= tinf_h else 0.0

    for i in range(steps):
        t = i * dt
        C1 = A1 / VC
        C2 = A2 / VP
        inf = infusion(t)

        dA1 = inf - CL * C1 - Q * (C1 - C2)
        dA2 = Q * (C1 - C2)

        A1 += dA1 * dt
        A2 += dA2 * dt

    return A1 / VC

def run_ppk(csv_path, save_output=True):
    df = pd.read_csv(csv_path)
    df = preprocess_dataframe(df)
    df["eGFR"] = np.nan
    df["CL_pop"] = np.nan
    df["Pred_C_pop"] = np.nan

    for i, row in df.iterrows():
        scr_mgdl = float(row["SCr"]) / 88.4
        eGFR = ckd_epi_2021(float(row["Age"]), float(row["Sex"]), scr_mgdl)
        df.loc[i, "eGFR"] = float(f"{eGFR:.2f}")
        tau_h = parse_frequency(row["Frequency"])

        dose_mg = float(row["Dru dose"]) * 1000
        tinf_h = float(row["Infusion time"]) / 60

        CL = THETA1_CL * (1 + THETA2_GFR * (eGFR - GFR_MEAN))
        if CL < 0.01: CL = 0.01
        df.loc[i, "CL_pop"] = float(f"{CL:.2f}")

        # prediciton
        pred = simulate_trough(dose_mg, tinf_h, tau_h, CL)
        df.loc[i, "Pred_C_pop"] = float(f"{pred:.2f}")

    if "C" in df.columns:
        valid = df[["C","Pred_C_pop"]].dropna()
        if len(valid) > 1:
            y_true = valid["C"].astype(float)
            y_pred = valid["Pred_C_pop"]
            R2 = r2_score(y_true,y_pred)
        else:
            R2 = np.nan
    else:
        R2 = np.nan

    print(f"\n POP-only R² = {R2:.4f}\n")

    if save_output:
        out_path = csv_path.replace(".csv", "_ppk_result.csv")
        df.to_csv(out_path, index=False)

    return df, R2

df, R2 = run_ppk("../data/X_val.csv")