
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

INPUT_FILE = (
    BASE_DIR
    / "evaluation_results"
    / "paired_analysis"
    / "paired_image_metrics.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "evaluation_results"
    / "statistical_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = ["Platelets", "RBC", "WBC"]

N_BOOTSTRAP = 10000
N_PERMUTATIONS = 100000

RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


# =========================================================
# 2. LOAD AND VALIDATE DATA
# =========================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

if len(df) != 36:
    print(
        f"WARNING: Expected 36 images, found {len(df)}."
    )

required_columns = ["image_id"]

for method in ["baseline", "qaps"]:
    for cls in CLASSES:
        for metric in ["tp", "fp", "fn"]:
            required_columns.append(
                f"{method}_{cls}_{metric}"
            )

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

if df["image_id"].duplicated().any():
    raise ValueError(
        "Duplicate image IDs detected."
    )

print("=" * 65)
print("PAIRED STATISTICAL ANALYSIS")
print("=" * 65)

print(f"Images evaluated: {len(df)}")


# =========================================================
# 3. CALCULATE PER-IMAGE METRICS
# =========================================================

def calculate_metrics(tp, fp, fn):

    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)

    precision_denominator = tp + fp
    recall_denominator = tp + fn

    precision = np.divide(
        tp,
        precision_denominator,
        out=np.zeros_like(tp),
        where=precision_denominator > 0
    )

    recall = np.divide(
        tp,
        recall_denominator,
        out=np.zeros_like(tp),
        where=recall_denominator > 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0
    )

    return precision, recall, f1


image_results = pd.DataFrame()

image_results["image_id"] = df["image_id"]


for method in ["baseline", "qaps"]:

    for cls in CLASSES:

        tp = df[f"{method}_{cls}_tp"].values
        fp = df[f"{method}_{cls}_fp"].values
        fn = df[f"{method}_{cls}_fn"].values

        precision, recall, f1 = calculate_metrics(
            tp, fp, fn
        )

        image_results[
            f"{method}_{cls}_precision"
        ] = precision

        image_results[
            f"{method}_{cls}_recall"
        ] = recall

        image_results[
            f"{method}_{cls}_f1"
        ] = f1


    # Per-image macro metrics:
    # Equal weight for each of the three classes.

    image_results[
        f"{method}_macro_precision"
    ] = image_results[
        [f"{method}_{c}_precision" for c in CLASSES]
    ].mean(axis=1)

    image_results[
        f"{method}_macro_recall"
    ] = image_results[
        [f"{method}_{c}_recall" for c in CLASSES]
    ].mean(axis=1)

    image_results[
        f"{method}_macro_f1"
    ] = image_results[
        [f"{method}_{c}_f1" for c in CLASSES]
    ].mean(axis=1)


image_results.to_csv(
    OUTPUT_DIR / "per_image_metrics.csv",
    index=False
)


# =========================================================
# 4. PAIRED PERMUTATION TEST
# =========================================================

def paired_permutation_test(baseline, qaps):

    baseline = np.asarray(baseline, dtype=float)
    qaps = np.asarray(qaps, dtype=float)

    differences = qaps - baseline

    observed = abs(np.mean(differences))

    extreme = 0
    completed = 0

    batch_size = 5000

    while completed < N_PERMUTATIONS:

        current_size = min(
            batch_size,
            N_PERMUTATIONS - completed
        )

        signs = rng.choice(
            [-1, 1],
            size=(current_size, len(differences))
        )

        permuted = np.mean(
            signs * differences,
            axis=1
        )

        extreme += np.sum(
            np.abs(permuted) >= observed
        )

        completed += current_size

    p_value = (
        extreme + 1
    ) / (
        N_PERMUTATIONS + 1
    )

    return p_value


# =========================================================
# 5. PAIRED BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================

def paired_bootstrap_ci(baseline, qaps):

    baseline = np.asarray(baseline, dtype=float)
    qaps = np.asarray(qaps, dtype=float)

    differences = qaps - baseline

    n = len(differences)

    bootstrap_means = []

    for _ in range(N_BOOTSTRAP):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        sample = differences[indices]

        bootstrap_means.append(
            np.mean(sample)
        )

    lower, upper = np.percentile(
        bootstrap_means,
        [2.5, 97.5]
    )

    return lower, upper


# =========================================================
# 6. EFFECT SIZE
# =========================================================

def cohens_dz(baseline, qaps):

    differences = (
        np.asarray(qaps, dtype=float)
        - np.asarray(baseline, dtype=float)
    )

    standard_deviation = np.std(
        differences,
        ddof=1
    )

    if standard_deviation == 0:
        return 0.0

    return (
        np.mean(differences)
        / standard_deviation
    )


# =========================================================
# 7. HOLM MULTIPLE-COMPARISON CORRECTION
# =========================================================

def holm_correction(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(p_values)

    order = np.argsort(p_values)

    adjusted = np.empty(n)

    running_max = 0.0

    for rank, index in enumerate(order):

        adjusted_value = (
            (n - rank) * p_values[index]
        )

        running_max = max(
            running_max,
            adjusted_value
        )

        adjusted[index] = min(
            running_max,
            1.0
        )

    return adjusted


# =========================================================
# 8. DEFINE PRIMARY AND SECONDARY COMPARISONS
# =========================================================

comparisons = [
    (
        "Macro F1",
        "baseline_macro_f1",
        "qaps_macro_f1"
    )
]

for cls in CLASSES:

    comparisons.append(
        (
            f"{cls} F1",
            f"baseline_{cls}_f1",
            f"qaps_{cls}_f1"
        )
    )


# =========================================================
# 9. RUN STATISTICAL TESTS
# =========================================================

results = []

for name, baseline_col, qaps_col in comparisons:

    baseline = image_results[
        baseline_col
    ].values

    qaps = image_results[
        qaps_col
    ].values

    differences = qaps - baseline

    if np.allclose(differences, 0):

        wilcoxon_p = 1.0

    else:

        wilcoxon_result = wilcoxon(
            qaps,
            baseline,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto"
        )

        wilcoxon_p = wilcoxon_result.pvalue


    permutation_p = paired_permutation_test(
        baseline,
        qaps
    )

    ci_lower, ci_upper = paired_bootstrap_ci(
        baseline,
        qaps
    )

    effect_size = cohens_dz(
        baseline,
        qaps
    )

    results.append({

        "Metric": name,

        "Baseline_Mean": np.mean(baseline),

        "QAPS_Mean": np.mean(qaps),

        "Mean_Difference": np.mean(differences),

        "Median_Difference": np.median(differences),

        "CI_95_Lower": ci_lower,

        "CI_95_Upper": ci_upper,

        "Wilcoxon_P_Value": wilcoxon_p,

        "Permutation_P_Value": permutation_p,

        "Cohens_Dz": effect_size,

        "Images_Improved": int(
            np.sum(differences > 0)
        ),

        "Images_Unchanged": int(
            np.sum(differences == 0)
        ),

        "Images_Worsened": int(
            np.sum(differences < 0)
        )

    })


# =========================================================
# 10. MULTIPLE COMPARISON CORRECTION
# =========================================================

results_df = pd.DataFrame(results)

results_df[
    "Permutation_P_Holm"
] = holm_correction(
    results_df["Permutation_P_Value"].values
)

results_df[
    "Wilcoxon_P_Holm"
] = holm_correction(
    results_df["Wilcoxon_P_Value"].values
)

results_df["Significant_0.05"] = (
    results_df["Permutation_P_Holm"] < 0.05
)


# =========================================================
# 11. SAVE RESULTS
# =========================================================

results_df.to_csv(
    OUTPUT_DIR / "statistical_results.csv",
    index=False
)

results_df.to_json(
    OUTPUT_DIR / "statistical_results.json",
    orient="records",
    indent=4
)


# =========================================================
# 12. DISPLAY RESULTS
# =========================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

print("\nSTATISTICAL RESULTS")
print("=" * 65)

print(
    results_df.round(6).to_string(
        index=False
    )
)

print("\nIMAGE-LEVEL METRICS SUMMARY")
print("=" * 65)

print(
    image_results.drop(
        columns=["image_id"]
    ).describe().round(4)
)

print("\nSTATISTICAL ANALYSIS COMPLETED")

print(
    f"\nResults saved in: {OUTPUT_DIR}"
)
