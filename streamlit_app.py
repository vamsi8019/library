import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st


def _load_rfid_library_module():
    module_path = Path(__file__).with_name("rfid_library_management.py")
    spec = importlib.util.spec_from_file_location("rfid_library_management_local", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load helper module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_library = _load_rfid_library_module()
detect_lost_books = _library.detect_lost_books
generate_synthetic_transactions = _library.generate_synthetic_transactions
predict_due_date_violations = _library.predict_due_date_violations
prepare_availability_series = _library.prepare_availability_series
train_availability_models = _library.train_availability_models
train_demand_model = _library.train_demand_model


st.set_page_config(page_title="RFID Library Management Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def run_pipeline(total_rows: int, overdue_threshold_days: int):
    transactions, book_df = generate_synthetic_transactions(total_rows=total_rows)
    demand_test, demand_metrics = train_demand_model(transactions)
    availability_test, availability_metrics = train_availability_models(
        prepare_availability_series(transactions, book_df)
    )
    lost_books = detect_lost_books(transactions, overdue_threshold_days=overdue_threshold_days)
    due_predictions, due_model_accuracy = predict_due_date_violations(transactions)
    return (
        transactions,
        demand_test,
        demand_metrics,
        availability_test,
        availability_metrics,
        lost_books,
        due_predictions,
        due_model_accuracy,
    )


def render_dashboard(transactions, demand_test, availability_test, lost_books, due_predictions):
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))

    # 1) Book demand prediction: compare model output against real monthly demand.
    ax = axes[0, 0]
    ax.plot(demand_test["borrow_date"], demand_test["actual_demand"], marker="o", label="Actual")
    ax.plot(
        demand_test["borrow_date"],
        demand_test["predicted_demand"],
        marker="o",
        linestyle="--",
        label="Predicted",
    )
    ax.set_title("Book Demand Prediction (Actual vs Predicted)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Borrows")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    # 2) Most popular books: top books ranked by borrow count.
    top_books = (
        transactions.groupby("book_title")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .sort_values(ascending=True)
    )
    ax = axes[0, 1]
    ax.barh(top_books.index, top_books.values, color="#2a9d8f")
    ax.set_title("Top 10 Most Popular Books")
    ax.set_xlabel("Borrow Count")
    ax.set_ylabel("Book Title")

    # 3) Peak usage time: hourly checkout pattern extracted from transactions.
    hourly = transactions.groupby("checkout_hour").size().reindex(range(8, 21), fill_value=0)
    ax = axes[1, 0]
    ax.plot(hourly.index, hourly.values, marker="o", color="#e76f51")
    ax.set_title("Peak Usage Time (Hourly Checkouts)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Checkouts")

    # 4) Availability prediction: compare three models against actual available stock.
    ax = axes[1, 1]
    ax.plot(availability_test["date"], availability_test["available_books"], label="Actual", linewidth=2)
    ax.plot(availability_test["date"], availability_test["pred_linear"], linestyle="--", label="Linear Regression")
    ax.plot(
        availability_test["date"],
        availability_test["pred_holt_winters"],
        linestyle="--",
        label="Holt-Winters",
    )
    ax.plot(availability_test["date"], availability_test["pred_tree"], linestyle="--", label="Decision Tree")
    ax.set_title("Availability Prediction Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Available Books")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    # 5) Lost book detection: show category-wise count of suspicious not-returned books.
    ax = axes[2, 0]
    if lost_books.empty:
        ax.text(0.5, 0.5, "No lost books detected", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        lost_counts = lost_books.groupby("book_category").size().sort_values(ascending=False)
        ax.bar(lost_counts.index, lost_counts.values, color="#d62828")
        ax.set_title("Lost Book Detection (Flagged Active Loans)")
        ax.set_xlabel("Category")
        ax.set_ylabel("Lost Candidates")

    # 6) Due-date violation prediction: highest-risk active loans based on classifier probability.
    ax = axes[2, 1]
    if due_predictions.empty:
        ax.text(0.5, 0.5, "No active borrows for due-date prediction", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        show = due_predictions.head(10).copy()
        show["label"] = show["book_title"].str.slice(0, 22) + " | " + show["user_role"].str[0].str.upper()
        show = show.sort_values("late_probability")
        ax.barh(show["label"], show["late_probability"], color="#6a4c93")
        ax.set_title("Predicted Due-Date Violation Risk (Top Active Loans)")
        ax.set_xlabel("Predicted Late Probability")
        ax.set_ylabel("Book | Role")

    fig.tight_layout()
    return fig


def main():
    st.title("RFID Library Management System")
    st.caption("Synthetic data + prediction models + single analytics dashboard")

    with st.sidebar:
        st.header("Controls")
        total_rows = st.slider("Number of transactions", min_value=1000, max_value=5000, value=1000, step=500)
        overdue_threshold_days = st.slider("Lost-book overdue threshold (days)", min_value=7, max_value=45, value=14)

    (
        transactions,
        demand_test,
        demand_metrics,
        availability_test,
        availability_metrics,
        lost_books,
        due_predictions,
        due_model_accuracy,
    ) = run_pipeline(total_rows=total_rows, overdue_threshold_days=overdue_threshold_days)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Demand MAE", f"{demand_metrics.mae:.3f}")
    m2.metric("Demand R²", f"{demand_metrics.r2:.3f}")
    m3.metric("Holt-Winters MAE", f"{availability_metrics['holt_winters'].mae:.3f}")
    m4.metric("Holt-Winters R²", f"{availability_metrics['holt_winters'].r2:.3f}")
    m5.metric("Due Violation Acc", f"{due_model_accuracy:.3f}")

    st.subheader("All Visualizations in One Dashboard")
    fig = render_dashboard(transactions, demand_test, availability_test, lost_books, due_predictions)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.subheader("Model Metrics (MAE, R²)")
    st.dataframe(
        {
            "Model": [
                demand_metrics.name,
                availability_metrics["linear"].name,
                availability_metrics["holt_winters"].name,
                availability_metrics["tree"].name,
            ],
            "MAE": [
                demand_metrics.mae,
                availability_metrics["linear"].mae,
                availability_metrics["holt_winters"].mae,
                availability_metrics["tree"].mae,
            ],
            "R2": [
                demand_metrics.r2,
                availability_metrics["linear"].r2,
                availability_metrics["holt_winters"].r2,
                availability_metrics["tree"].r2,
            ],
        },
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Lost Book Candidates")
    if lost_books.empty:
        st.info("No lost books detected with current threshold.")
    else:
        st.dataframe(
            lost_books[["user_id", "book_title", "borrow_date", "borrow_age", "book_category"]].head(20),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Predicted Due-Date Violations (Active Borrows)")
    if due_predictions.empty:
        st.info("No active borrows available for due-date prediction.")
    else:
        st.dataframe(
            due_predictions[
                [
                    "user_id",
                    "book_title",
                    "borrow_date",
                    "late_probability",
                    "predicted_late",
                ]
            ].head(20),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
