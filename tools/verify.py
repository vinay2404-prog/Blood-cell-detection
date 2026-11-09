
import pandas as pd

df = pd.read_csv(
    "evaluation_results/paired_analysis/paired_image_metrics.csv"
)

for cls in ["Platelets", "RBC", "WBC"]:

    print(f"\n{cls}")

    for method in ["baseline", "qaps"]:

        print(
            method,
            {
                "TP": int(df[f"{method}_{cls}_tp"].sum()),
                "FP": int(df[f"{method}_{cls}_fp"].sum()),
                "FN": int(df[f"{method}_{cls}_fn"].sum())
            }
        )

    if cls == "Platelets":

        print("\nPlatelet FP changes:")

        changes = df[
            [
                "image_id",
                "baseline_Platelets_fp",
                "qaps_Platelets_fp"
            ]
        ].copy()

        changes["FP_difference"] = (
            changes["qaps_Platelets_fp"]
            - changes["baseline_Platelets_fp"]
        )

        print(
            changes[
                changes["FP_difference"] != 0
            ].to_string(index=False)
        )

        print(
            "\nTotal FP difference:",
            changes["FP_difference"].sum()
        )
