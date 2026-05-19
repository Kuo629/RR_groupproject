from pathlib import Path
import re
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

RAW_DATA_PATH = Path("data/raw/loan_data.csv")

PROCESSED_DIR = Path("data/processed")
TABLES_DIR = Path("output/tables")
FIGURES_DIR = Path("output/figures")

CLEAN_DATA_PATH = PROCESSED_DIR / "loan_clean.csv"
MODEL_READY_DATA_PATH = PROCESSED_DIR / "loan_model_ready.csv"


# ============================================================
# R-to-Python translation notes
# ============================================================
# This script reproduces the data loading, type fixing, data audit,
# EDA, and cleaning pipeline from the original R workflow:
#
# R:
#   read_csv("data/raw/loan_data.csv") %>% clean_names()
#   mutate(... factor conversions ...)
#   glimpse(df), summary(df), n_distinct(), missingness scan
#   numerical histograms, boxplots, correlation matrix
#   business-rule cleaning
#   winsorization at 99.5th percentile
#   drop_na() on critical model fields
#
# Python:
#   pandas read_csv()
#   snake_case column cleaning
#   categorical / integer type fixes
#   data summaries and EDA outputs
#   same business-rule cleaning
#   same 99.5th percentile capping
#   drop rows missing critical fields


# ============================================================
# Helpers
# ============================================================

def clean_column_names(columns: list[str]) -> list[str]:
    """
    Reproduce janitor::clean_names() approximately:
    lowercase names and convert non-alphanumeric characters to underscores.
    """
    clean_cols = []
    for col in columns:
        col = col.strip().lower()
        col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
        col = re.sub(r"_+", "_", col)
        col = col.strip("_")
        clean_cols.append(col)
    return clean_cols


def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    """
    Load raw loan data from the same repo path used in the R workflow.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            "Please place loan_data.csv in data/raw/."
        )

    df = pd.read_csv(path)
    df.columns = clean_column_names(df.columns.tolist())
    return df


def apply_type_fixes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the R mutate() type-fixing step.

    R converts selected variables to factor and converts selected
    integer-like variables to integer. It also converts loan_status
    into factor labels: Denied / Approved.
    """
    df_fixed = df.copy()

    categorical_cols = [
        "person_gender",
        "person_education",
        "person_home_ownership",
        "loan_intent",
        "previous_loan_defaults_on_file",
    ]

    for col in categorical_cols:
        if col in df_fixed.columns:
            df_fixed[col] = df_fixed[col].astype("category")

    # R: factor(loan_status, levels = c(0, 1), labels = c("Denied", "Approved"))
    if "loan_status" in df_fixed.columns:
        df_fixed["loan_status"] = df_fixed["loan_status"].map({
            0: "Denied",
            1: "Approved"
        }).astype("category")

    # R optional integer-ish cleanup
    integer_like_cols = [
        "person_emp_exp",
        "cb_person_cred_hist_length",
        "person_age",
    ]

    for col in integer_like_cols:
        if col in df_fixed.columns:
            df_fixed[col] = pd.to_numeric(df_fixed[col], errors="coerce").astype("Int64")

    return df_fixed


def audit_data(df: pd.DataFrame, stage_name: str) -> None:
    """
    Reproduce R glimpse(), summary(), n_distinct(),
    missingness scan, and target distribution.
    """
    print(f"\n================ {stage_name} ================")

    print("\nRows:", len(df))
    print("Columns:", df.shape[1])

    print("\n=== Column Names ===")
    print(df.columns.tolist())

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Summary ===")
    print(df.describe(include="all"))

    print("\n=== Distinct Value Counts ===")
    print(df.nunique(dropna=False))

    print("\n=== Missing Values ===")
    print(df.isna().sum().sort_values(ascending=False))

    if "loan_status" in df.columns:
        print("\n=== Target Distribution ===")
        print(df["loan_status"].value_counts(dropna=False))
        print("\n=== Target Distribution Percentage ===")
        print(df["loan_status"].value_counts(normalize=True, dropna=False).round(3))


def save_audit_tables(df: pd.DataFrame, stage_name: str) -> None:
    """Save audit outputs as CSV tables."""
    prefix = stage_name.lower().replace(" ", "_")

    missing_values = df.isna().sum().reset_index()
    missing_values.columns = ["variable", "missing_count"]
    missing_values = missing_values.sort_values("missing_count", ascending=False)
    missing_values.to_csv(TABLES_DIR / f"{prefix}_missing_values.csv", index=False)

    distinct_values = df.nunique(dropna=False).reset_index()
    distinct_values.columns = ["variable", "distinct_count"]
    distinct_values.to_csv(TABLES_DIR / f"{prefix}_distinct_values.csv", index=False)

    if "loan_status" in df.columns:
        target_distribution = df["loan_status"].value_counts(dropna=False).reset_index()
        target_distribution.columns = ["loan_status", "count"]
        target_distribution["percentage"] = (
            target_distribution["count"] / target_distribution["count"].sum()
        ).round(3)
        target_distribution.to_csv(TABLES_DIR / f"{prefix}_target_distribution.csv", index=False)


def apply_business_rule_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the R business-rule cleaning step.

    R logic:
    - person_age < 18 or > 100 -> NA
    - person_emp_exp < 0 or > 60 -> NA
    - cb_person_cred_hist_length < 0 or > 60 -> NA
    - credit_score < 300 or > 850 -> NA
    - loan_int_rate < 0 or > 60 -> NA
    - loan_percent_income < 0 or > 1.5 -> NA
    - person_income <= 0 -> NA
    - loan_amnt <= 0 -> NA
    """
    df_clean = df.copy()

    rules = {
        "person_age": lambda s: (s < 18) | (s > 100),
        "person_emp_exp": lambda s: (s < 0) | (s > 60),
        "cb_person_cred_hist_length": lambda s: (s < 0) | (s > 60),
        "credit_score": lambda s: (s < 300) | (s > 850),
        "loan_int_rate": lambda s: (s < 0) | (s > 60),
        "loan_percent_income": lambda s: (s < 0) | (s > 1.5),
        "person_income": lambda s: s <= 0,
        "loan_amnt": lambda s: s <= 0,
    }

    for col, invalid_rule in rules.items():
        if col in df_clean.columns:
            numeric_col = pd.to_numeric(df_clean[col], errors="coerce")
            invalid_mask = invalid_rule(numeric_col)
            df_clean.loc[invalid_mask, col] = np.nan

    return df_clean


def winsorize_heavy_tails(df: pd.DataFrame, percentile: float = 0.995) -> pd.DataFrame:
    """
    Reproduce the R cap() function:
    cap person_income and loan_amnt at the 99.5th percentile.
    """
    df_capped = df.copy()

    for col in ["person_income", "loan_amnt"]:
        if col in df_capped.columns:
            upper_cap = df_capped[col].quantile(percentile)
            df_capped[col] = np.minimum(df_capped[col], upper_cap)

    return df_capped


def drop_critical_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce R drop_na() on critical model fields:
    loan_status, person_age, person_income, person_emp_exp,
    loan_amnt, credit_score, loan_int_rate.
    """
    critical_fields = [
        "loan_status",
        "person_age",
        "person_income",
        "person_emp_exp",
        "loan_amnt",
        "credit_score",
        "loan_int_rate",
    ]

    existing_critical_fields = [col for col in critical_fields if col in df.columns]

    return df.dropna(subset=existing_critical_fields).copy()


def save_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Reproduce numeric-only correlation matrix from the R workflow.
    """
    numeric_df = df.select_dtypes(include=["number"])
    correlation_matrix = numeric_df.corr().round(2)
    correlation_matrix.to_csv(TABLES_DIR / "correlation_matrix.csv")


def plot_numeric_histograms(df: pd.DataFrame) -> None:
    """
    Reproduce R numeric histogram faceting in Python.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=40)
        axes[i].set_title(col)
        axes[i].set_xlabel("value")
        axes[i].set_ylabel("count")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_histograms.png", dpi=300)
    plt.close(fig)


def plot_numeric_boxplots(df: pd.DataFrame) -> None:
    """
    Reproduce R numeric boxplot faceting in Python.
    """
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        return

    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(numeric_cols):
        axes[i].boxplot(df[col].dropna(), vert=True)
        axes[i].set_title(col)
        axes[i].set_ylabel("value")

    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "numeric_boxplots.png", dpi=300)
    plt.close(fig)


def plot_target_distribution(df: pd.DataFrame) -> None:
    """
    Save a simple target distribution plot.
    This extends the R table output with a visual figure.
    """
    if "loan_status" not in df.columns:
        return

    counts = df["loan_status"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Loan Status Distribution")
    ax.set_xlabel("loan_status")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=300)
    plt.close(fig)


def create_model_ready_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a Python model-ready version of the cleaned dataset.

    Important:
    - R randomForest/rpart can handle factor variables directly.
    - scikit-learn models generally need numeric features.
    - Therefore, one-hot encoding is a Python adaptation for modelling,
      not a separate R cleaning rule.
    """
    df_model = df.copy()

    if "loan_status" not in df_model.columns:
        raise ValueError("loan_status column is missing.")

    # Convert target back to binary for Python modelling.
    target_binary = df_model["loan_status"].map({
        "Denied": 0,
        "Approved": 1
    }).astype(int)

    features = df_model.drop(columns=["loan_status"])
    categorical_cols = features.select_dtypes(include=["object", "category"]).columns.tolist()

    features_encoded = pd.get_dummies(
        features,
        columns=categorical_cols,
        drop_first=False,
        dtype=int
    )

    df_model_ready = features_encoded.copy()
    df_model_ready["loan_status"] = target_binary.values

    return df_model_ready


def main() -> None:
    ensure_output_dirs()

    # 1. Load data and clean column names, matching R read_csv() %>% clean_names()
    df_raw = load_data(RAW_DATA_PATH)

    # 2. Type fixes, matching R mutate(... factor ...)
    df_typed = apply_type_fixes(df_raw)

    # 3. Audit raw typed data, matching R glimpse(), summary(), n_distinct(), missing scan
    audit_data(df_typed, "Raw Typed Data")
    save_audit_tables(df_typed, "Raw Typed Data")

    # 4. EDA on raw typed data, matching R numeric histograms, boxplots, correlations
    plot_numeric_histograms(df_typed)
    plot_numeric_boxplots(df_typed)
    plot_target_distribution(df_typed)
    save_correlation_matrix(df_typed)

    # 5. Business-rule sanity cleaning, matching R df_clean_step1
    df_clean_step1 = apply_business_rule_cleaning(df_typed)

    # 6. Winsorization, matching R df_clean_step2
    df_clean_step2 = winsorize_heavy_tails(df_clean_step1, percentile=0.995)

    # 7. Drop critical missing rows, matching R df_clean
    df_clean = drop_critical_missing_rows(df_clean_step2)

    # 8. Audit cleaned data
    audit_data(df_clean, "Clean Data")
    save_audit_tables(df_clean, "Clean Data")

    # 9. Save cleaned R-equivalent data
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)

    # 10. Save Python model-ready encoded data
    df_model_ready = create_model_ready_data(df_clean)
    df_model_ready.to_csv(MODEL_READY_DATA_PATH, index=False)

    # 11. Save cleaning report
    cleaning_report = pd.DataFrame({
        "stage": ["raw_typed_data", "clean_data"],
        "rows": [len(df_typed), len(df_clean)],
        "columns": [df_typed.shape[1], df_clean.shape[1]],
    })
    cleaning_report.to_csv(TABLES_DIR / "cleaning_report.csv", index=False)

    print("\n================ Saved Outputs ================")
    print(f"Clean data saved to: {CLEAN_DATA_PATH}")
    print(f"Model-ready data saved to: {MODEL_READY_DATA_PATH}")
    print(f"Tables saved to: {TABLES_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
