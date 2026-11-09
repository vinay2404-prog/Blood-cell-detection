
import pandas as pd
import numpy as np

df = pd.read_csv(
    "evaluation_results/paired_analysis/paired_image_metrics.csv"
)

classes = ["Platelets", "RBC", "WBC"]

print("=" * 65)
print("VERIFYING PAIRED AND STATISTICAL ANALYSIS CONSISTENCY")
print("=" * 65)

for cls in classes:

    print(f"\nCLASS: {cls}")

    for method in ["baseline", "qaps"]:

        tp = df[f"{method}_{cls}_tp"]
        fp = df[f"{method}_{cls}_fp"]
        fn = df[f"{method}_{cls}_fn"]

        precision = np.divide(
            tp,
            tp + fp,
            out=np.zeros(len(df), dtype=float),
            where=(tp + fp) > 0
        )

        recall = np.divide(
            tp,
            tp + fn,
            out=np.zeros(len(df), dtype=float),
            where=(tp + fn) > 0
        )

        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(len(df), dtype=float),
            where=(precision + recall) > 0
        )

        print(f"\n{method.upper()}")
        print("Total TP:", tp.sum())
        print("Total FP:", fp.sum())
        print("Total FN:", fn.sum())
        print("Pooled F1:",
              2 * tp.sum() /
              (2 * tp.sum() + fp.sum() + fn.sum()))
        print("Mean image F1:", f1.mean())

        df[f"{method}_{cls}_verified_f1"] = f1

    baseline = df[f"baseline_{cls}_verified_f1"]
    qaps = df[f"qaps_{cls}_verified_f1"]

    difference = qaps - baseline

    print("\nPAIRED IMAGE COMPARISON")
    print("Mean difference:", difference.mean())
    print("Improved:", (difference > 1e-12).sum())
    print("Unchanged:", np.isclose(difference, 0).sum())
    print("Worsened:", (difference < -1e-12).sum())

    changed = df.loc[
        ~np.isclose(difference, 0),
        ["image_id",
         f"baseline_{cls}_tp",
         f"baseline_{cls}_fp",
         f"baseline_{cls}_fn",
         f"qaps_{cls}_tp",
         f"qaps_{cls}_fp",
         f"qaps_{cls}_fn"]
    ].copy()

    if not changed.empty:
        print("\nImages with changed counts:")
        print(changed.to_string(index=False))

print("\nVERIFICATION COMPLETED")
