import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import r2_score

cnn_val_pred = pd.read_csv("../data/cnn_val_result.csv", index_col=0)
ml_val_pred  = pd.read_csv("../data/ml_val_result.csv", index_col=0)

true_y = cnn_val_pred[["True_Y"]].copy()
df_mlp = cnn_val_pred.drop(columns=["True_Y"])
df_ml  = ml_val_pred.drop(columns=["True_Y"])
combined_df = pd.concat([df_mlp, df_ml, true_y], axis=1)
cols = [col for col in combined_df.columns if col != "True_Y"] + ["True_Y"]
combined_df = combined_df[cols]


def ensemble_negative_to_zero_and_find_best(combined_df):
    y_true = combined_df['True_Y']
    models = [col for col in combined_df.columns if col != "True_Y"]
    df_non_negative = combined_df.copy()
    df_non_negative[models] = df_non_negative[models].clip(lower=0)

    print("\n📌 single model R2：")
    single_r2 = {}
    for model in models:
        r2 = r2_score(y_true, df_non_negative[model])
        single_r2[model] = r2
        print(f"{model}: {r2:.4f}")

    best_r2 = -np.inf
    best_combination = None
    results = []

    for r in range(1, len(models) + 1):
        for combo in combinations(models, r):
            pred_matrix = df_non_negative[list(combo)]
            ensemble_pred = pred_matrix.mean(axis=1)
            r2 = r2_score(y_true, ensemble_pred)
            results.append((combo, r2))

            if r2 > best_r2:
                best_r2 = r2
                best_combination = combo

    print("\n🎯 the best combinations")
    print(" + ".join(best_combination))
    print(f"R2 = {best_r2:.4f}")

    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    for combo, r2 in results_sorted[:20]:
        print(f"{combo} → R2={r2:.4f}")

    best_pred = df_non_negative[list(best_combination)].mean(axis=1)
    pred_dict = {
        'y_true': y_true.values,
        'y_pred_best': best_pred.values
    }

    ensemble_name = "Ensemble_Model"
    combined_df[ensemble_name] = best_pred.values
    return (
        best_combination,
        best_r2,
        results_sorted,
        df_non_negative,
        pred_dict,
        combined_df 
    )

best_r2, best_comb, df_processed, single_r2_scores, pred_dict, combined_df_result = \
    ensemble_negative_to_zero_and_find_best(combined_df)

combined_df.to_csv("../data/model_results.csv")
