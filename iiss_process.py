from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.response_utils import get_response


datasets_dir = Path(__file__).parent / "datasets"
datasets_dir.mkdir(parents=True, exist_ok=True)

dataset_file = datasets_dir / "online_retail_dataset.csv"

if dataset_file.exists():
    print(f"Loading dataset from {dataset_file}...")
    df = pd.read_csv(dataset_file)
else:
    print("Dataset not found locally. Fetching from UCI repository...")
    from ucimlrepo import fetch_ucirepo

    online_retail = fetch_ucirepo(id=352)  # Online Retail Dataset
    df = online_retail.data.features  # type: ignore
    df.to_csv(dataset_file, index=False)
    print(f"Dataset saved to {dataset_file}")

# prompt = f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns. Summarize potential business insights."
# llm_response = get_response(prompt)


# ---- Data Ingestion & Preprocessing ----
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df["TotalValue"] = df["Quantity"] * df["UnitPrice"]

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

df = preprocess_data(df)


# ---- User Query Interpretation ----
scope_prompt = """
You are an analytical strategist. The executive asked: '{query}'.
Define the data analysis scope, metrics to evaluate, and potential modeling strategies.
"""

def analyze_user_query(query: str):
    """
    Interprets user/executive queries and defines the analysis scope.
    """
    print("\n[1] User Query Analysis")
    print(f"Executive query: {query}\n")

    analysis_scope = get_response(scope_prompt.format(query=query))
    return analysis_scope


# ---- Data Analytics & Modeling ----
def perform_data_analysis(df: pd.DataFrame):
    """
    Performs exploratory data analysis (EDA) and basic modeling.
    """
    print("\n[3] Data Analytics & Modeling")

    # Basic EDA summaries
    summary = {
        "total_revenue": df["TotalValue"].sum(),
        "unique_customers": df["CustomerID"].nunique(),
        "unique_products": df["StockCode"].nunique(),
        "countries": df["Country"].nunique(),
        "top_country": df["Country"].value_counts().idxmax(),
    }

    print("Key EDA Metrics:", summary)

    # Revenue by country
    top_countries = df.groupby("Country")["TotalValue"].sum().sort_values(ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_countries.values, y=top_countries.index)
    plt.title("Top 10 Countries by Total Revenue")
    plt.xlabel("Total Revenue")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()

    # Modeling: Customer Segmentation (RFM)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg({
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "count",
            "TotalValue": "sum"
        })
        .rename(columns={
            "InvoiceDate": "Recency",
            "InvoiceNo": "Frequency",
            "TotalValue": "Monetary"
        })
    )

    # Simple scoring
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["RFM_Score"] = rfm[["R_Score", "F_Score", "M_Score"]].sum(axis=1)

    return summary, rfm


# ---- Validation & Review ----
review_prompt = """
Validate the following analysis results for business soundness:

Summary: {summary}
Sample RFM Data:
{rfm_data}

Are the findings logical and useful for executive decision-making?
Suggest refinements if needed.
"""
def validate_results(summary: str, rfm: pd.DataFrame):
    """
    Validates insights and requests peer review (through LLM check).
    """
    print("\n[4] Validation & Peer Review")
    validation_feedback = get_response(review_prompt.format(
        summary=summary, rfm_data=rfm.head().to_string()
    ))
    return validation_feedback


# ---- Reporting & Visualization ----
report_prompt = """
Summarize the validated findings into a clear executive report.

Key metrics: {summary}
Feedback from peer review: {validation_feedback}

Create a concise narrative with recommendations for increasing revenue and customer retention.
"""
def generate_report(summary: str, validation_feedback: str):
    """
    Generates the final strategic report for the executive.
    """
    print("\n[5] Reporting & Visualization")

    report = get_response(report_prompt.format(
        summary=summary, validation_feedback=validation_feedback
    ))

    print("\n📈 Final Executive Report:")
    print(report)
    return report


# ---- Main IISS Pipeline ----
def run_iiss_pipeline(executive_query: str):
    analysis_scope = analyze_user_query(executive_query)

    summary, rfm = perform_data_analysis(df)
    summary_str = "\n".join([f"{k}: {v}" for k, v in summary.items()])

    validation_feedback = validate_results(summary_str, rfm)

    report = generate_report(summary_str, validation_feedback)

    return {
        "analysis_scope": analysis_scope,
        "summary": summary,
        "validation_feedback": validation_feedback,
        "report": report
    }


if __name__ == "__main__":
    executive_query = "How can we increase overall revenue by improving customer retention and targeting high-value clients?"
    iiss_output = run_iiss_pipeline(executive_query)
    print("\nIISS Pipeline Output:", iiss_output)
