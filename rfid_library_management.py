import uuid
from dataclasses import dataclass
from typing import Dict, Tuple
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
TODAY = pd.Timestamp("2026-04-13")


@dataclass
class ModelMetrics:
    name: str
    mae: float
    r2: float


def build_seed_rows() -> pd.DataFrame:
    """Start from user-provided sample rows to keep generated data grounded."""
    seed_rows = [
        (
            "d166c2c3-a24a-482f-bd50-c1c2b115aea4",
            "ed728782-79ea-4cec-ad23-236e745caff6",
            "13-03-2023",
            "29-03-2023",
            True,
            1,
            "student",
            "non-fiction",
        ),
        (
            "902f3b25-7d14-4943-8aff-27b3eac9a7aa",
            "02903451-e16b-49e3-8afc-69cba817208d",
            "02-04-2022",
            "07-04-2022",
            True,
            0,
            "student",
            "art",
        ),
        (
            "6cee8789-3fbe-4b35-bcf5-a891484adefe",
            "ba2eacff-aa53-4950-bd8d-5fe7d8a4db49",
            "06-06-2023",
            "08-06-2023",
            True,
            0,
            "faculty",
            "history",
        ),
        (
            "588e367e-1a23-41dc-a8a0-127a482c2e6a",
            "6a350522-8b9a-4581-aaa8-48e8ec90d997",
            "05-10-2022",
            "24-10-2022",
            False,
            1,
            "student",
            "non-fiction",
        ),
        (
            "3d15c259-fced-4210-9d6c-28292e9ded79",
            "1bcb4fa7-7e2c-41f9-8d6a-73e7c06db0ce",
            "04-07-2020",
            "09-07-2020",
            False,
            0,
            "student",
            "fiction",
        ),
        (
            "08152cfd-e6f1-4e73-a724-cfcc834aa6f4",
            "3b88f23b-acea-47a0-b152-fef8c15cd092",
            "28-10-2023",
            "15-11-2023",
            False,
            1,
            "student",
            "art",
        ),
        (
            "1978fe94-f10d-4f02-a646-67feeb3e6e04",
            "d236345c-2c29-4563-b468-864cb04db7d0",
            "11-04-2023",
            "30-04-2023",
            False,
            1,
            "staff",
            "science",
        ),
        (
            "d96237eb-0852-4437-a533-b2391adb419d",
            "a0c97d67-f478-4656-8241-cf4be94598a1",
            "21-07-2023",
            "09-08-2023",
            False,
            1,
            "staff",
            "fiction",
        ),
        (
            "c39287df-44f5-48d9-9948-e3454008ccd0",
            "66e82aea-b2f9-465c-808f-75e3ca7e5ddd",
            "01-09-2022",
            "16-09-2022",
            True,
            1,
            "student",
            "non-fiction",
        ),
        (
            "0aa0b427-f3bc-4ed6-924d-25589bfdc5f1",
            "f068d1ee-c021-4785-bd39-a2761563dbc1",
            "10-07-2023",
            "16-07-2023",
            False,
            0,
            "student",
            "science",
        ),
    ]

    columns = [
        "user_id",
        "book_id",
        "borrow_date",
        "return_date",
        "reservation_status",
        "overdue_status",
        "user_role",
        "book_category",
    ]
    seed_df = pd.DataFrame(seed_rows, columns=columns)
    seed_df["borrow_date"] = pd.to_datetime(seed_df["borrow_date"], format="%d-%m-%Y")
    seed_df["return_date"] = pd.to_datetime(seed_df["return_date"], format="%d-%m-%Y")
    return seed_df


def create_book_catalog(seed_df: pd.DataFrame, target_books: int = 240) -> pd.DataFrame:
    categories = ["fiction", "non-fiction", "science", "history", "art"]
    adjectives = [
        "Hidden",
        "Ancient",
        "Modern",
        "Digital",
        "Quantum",
        "Silent",
        "Crimson",
        "Emerald",
        "Practical",
        "Applied",
    ]
    nouns = [
        "Archive",
        "Methods",
        "Lab",
        "Atlas",
        "Blueprint",
        "Stories",
        "Voyage",
        "Principles",
        "Workshop",
        "Chronicles",
    ]

    books = []
    for i in range(target_books):
        category = np.random.choice(categories)
        title = f"{np.random.choice(adjectives)} {np.random.choice(nouns)} {i + 1}"
        books.append(
            {
                "book_id": str(uuid.uuid4()),
                "book_title": title,
                "book_category": category,
                "total_copies": np.random.randint(2, 8),
            }
        )

    book_df = pd.DataFrame(books)

    # Ensure provided sample book IDs are preserved in the catalog.
    for _, row in seed_df.iterrows():
        book_id = row["book_id"]
        if book_id not in set(book_df["book_id"]):
            book_df = pd.concat(
                [
                    book_df,
                    pd.DataFrame(
                        [
                            {
                                "book_id": book_id,
                                "book_title": f"Reference Book {book_id[:8]}",
                                "book_category": row["book_category"],
                                "total_copies": np.random.randint(2, 8),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    return book_df.drop_duplicates(subset=["book_id"]).reset_index(drop=True)


def generate_synthetic_transactions(total_rows: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    seed_df = build_seed_rows()
    book_df = create_book_catalog(seed_df)

    roles = ["student", "faculty", "staff"]
    categories = ["fiction", "non-fiction", "science", "history", "art"]

    role_weights = [0.58, 0.22, 0.20]
    category_weights = [0.24, 0.24, 0.22, 0.17, 0.13]

    user_pool = {
        role: [str(uuid.uuid4()) for _ in range(220 if role == "student" else 120)]
        for role in roles
    }

    generated = []
    remaining = total_rows - len(seed_df)

    for _ in range(remaining):
        role = np.random.choice(roles, p=role_weights)
        category = np.random.choice(categories, p=category_weights)

        # Borrow dates spread over multiple years with realistic seasonality.
        random_day = np.random.randint(0, (TODAY - pd.Timestamp("2020-01-01")).days)
        borrow_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(random_day))

        role_expected_days = {"student": 14, "faculty": 21, "staff": 10}
        expected_days = role_expected_days[role]

        # Return delays create both on-time and overdue behavior.
        delay = int(np.random.normal(loc=0.5, scale=4.5))
        loan_length = max(1, expected_days + delay)

        is_returned = np.random.rand() < 0.9
        if is_returned:
            return_date = borrow_date + pd.Timedelta(days=loan_length)
            overdue_status = int(return_date > borrow_date + pd.Timedelta(days=expected_days))
        else:
            return_date = pd.NaT
            overdue_status = int(TODAY > borrow_date + pd.Timedelta(days=expected_days))

        reservation_status = bool(np.random.rand() < 0.45)

        category_books = book_df[book_df["book_category"] == category]
        chosen_book = category_books.sample(1, random_state=np.random.randint(1, 10_000)).iloc[0]

        generated.append(
            {
                "user_id": np.random.choice(user_pool[role]),
                "book_id": chosen_book["book_id"],
                "borrow_date": borrow_date,
                "return_date": return_date,
                "reservation_status": reservation_status,
                "overdue_status": overdue_status,
                "user_role": role,
                "book_category": category,
            }
        )

    transactions = pd.concat([seed_df, pd.DataFrame(generated)], ignore_index=True)

    # Synthetic checkout hour is used for peak usage analysis.
    role_hour_center = {"student": 14, "faculty": 10, "staff": 12}
    transactions["checkout_hour"] = transactions["user_role"].map(role_hour_center) + np.random.randint(
        -3, 4, size=len(transactions)
    )
    transactions["checkout_hour"] = transactions["checkout_hour"].clip(8, 20)

    transactions = transactions.merge(book_df[["book_id", "book_title", "total_copies"]], on="book_id", how="left")
    return transactions, book_df


def prepare_availability_series(df: pd.DataFrame, book_df: pd.DataFrame) -> pd.DataFrame:
    start_date = df["borrow_date"].min()
    end_date = max(df["borrow_date"].max(), df["return_date"].dropna().max())
    daily_index = pd.date_range(start=start_date, end=end_date, freq="D")

    checkouts = df.groupby("borrow_date").size().reindex(daily_index, fill_value=0)
    returns = df.dropna(subset=["return_date"]).groupby("return_date").size().reindex(daily_index, fill_value=0)

    outstanding = (checkouts - returns).cumsum()
    total_capacity = int(book_df["total_copies"].sum())
    available = np.maximum(total_capacity - outstanding, 0)

    availability_df = pd.DataFrame(
        {
            "date": daily_index,
            "checkouts": checkouts.values,
            "returns": returns.values,
            "available_books": available,
        }
    )
    return availability_df


def train_demand_model(df: pd.DataFrame) -> Tuple[pd.DataFrame, ModelMetrics]:
    monthly = (
        df.set_index("borrow_date")
        .resample("MS")
        .size()
        .reset_index(name="actual_demand")
    )

    monthly["time_idx"] = np.arange(len(monthly))
    monthly["month"] = monthly["borrow_date"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["month"] / 12)

    split = int(len(monthly) * 0.8)
    train_df = monthly.iloc[:split]
    test_df = monthly.iloc[split:].copy()

    X_train = train_df[["time_idx", "month_sin", "month_cos"]]
    y_train = train_df["actual_demand"]
    X_test = test_df[["time_idx", "month_sin", "month_cos"]]

    # Linear trend + seasonality baseline for demand forecasting.
    model = LinearRegression()
    model.fit(X_train, y_train)
    test_df["predicted_demand"] = model.predict(X_test)

    mae = mean_absolute_error(test_df["actual_demand"], test_df["predicted_demand"])
    r2 = r2_score(test_df["actual_demand"], test_df["predicted_demand"])

    metrics = ModelMetrics("Demand Linear Regression", mae, r2)
    return test_df, metrics


def train_availability_models(availability_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, ModelMetrics]]:
    data = availability_df.copy().reset_index(drop=True)
    data["time_idx"] = np.arange(len(data))
    data["dow"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month

    split = int(len(data) * 0.8)
    train_df = data.iloc[:split]
    test_df = data.iloc[split:].copy()

    X_train = train_df[["time_idx", "dow", "month", "checkouts", "returns"]]
    y_train = train_df["available_books"]
    X_test = test_df[["time_idx", "dow", "month", "checkouts", "returns"]]

    metrics: Dict[str, ModelMetrics] = {}

    # Model 1: linear regression captures global trend and simple seasonality.
    linear = LinearRegression()
    linear.fit(X_train, y_train)
    test_df["pred_linear"] = linear.predict(X_test)
    metrics["linear"] = ModelMetrics(
        "Availability Linear Regression",
        mean_absolute_error(test_df["available_books"], test_df["pred_linear"]),
        r2_score(test_df["available_books"], test_df["pred_linear"]),
    )

    # Model 2: Holt-Winters models level/trend/weekly seasonality as a time series.
    hw_model = ExponentialSmoothing(
        train_df["available_books"], trend="add", seasonal="add", seasonal_periods=7
    ).fit(optimized=True)
    test_df["pred_holt_winters"] = hw_model.forecast(len(test_df)).values
    metrics["holt_winters"] = ModelMetrics(
        "Availability Holt-Winters",
        mean_absolute_error(test_df["available_books"], test_df["pred_holt_winters"]),
        r2_score(test_df["available_books"], test_df["pred_holt_winters"]),
    )

    # Model 3: decision tree captures nonlinear interactions in checkout/return patterns.
    tree = DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED)
    tree.fit(X_train, y_train)
    test_df["pred_tree"] = tree.predict(X_test)
    metrics["tree"] = ModelMetrics(
        "Availability Decision Tree Regressor",
        mean_absolute_error(test_df["available_books"], test_df["pred_tree"]),
        r2_score(test_df["available_books"], test_df["pred_tree"]),
    )

    return test_df, metrics


def detect_lost_books(df: pd.DataFrame, overdue_threshold_days: int = 14) -> pd.DataFrame:
    role_expected_days = {"student": 14, "faculty": 21, "staff": 10}

    active = df[df["return_date"].isna()].copy()
    active["expected_days"] = active["user_role"].map(role_expected_days)
    active["borrow_age"] = (TODAY - active["borrow_date"]).dt.days

    # Lost-book heuristic: active loan older than expected period + threshold.
    active["lost_flag"] = active["borrow_age"] > (active["expected_days"] + overdue_threshold_days)
    return active[active["lost_flag"]].sort_values("borrow_age", ascending=False)


def predict_due_date_violations(df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
    role_expected_days = {"student": 14, "faculty": 21, "staff": 10}

    model_df = df.copy()
    model_df["expected_days"] = model_df["user_role"].map(role_expected_days)
    model_df["days_borrowed"] = (
        model_df["return_date"].fillna(TODAY) - model_df["borrow_date"]
    ).dt.days
    model_df["days_since_due"] = (model_df["days_borrowed"] - model_df["expected_days"]).clip(lower=0)
    model_df["reservation_status"] = model_df["reservation_status"].astype(int)
    model_df["violation_label"] = model_df["overdue_status"].astype(int)

    features = [
        "days_borrowed",
        "days_since_due",
        "reservation_status",
        "user_role",
        "book_category",
    ]

    X = model_df[features]
    y = model_df["violation_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["user_role", "book_category"]),
        ],
        remainder="passthrough",
    )

    # Classifier estimates probability that a currently borrowed book will violate due date.
    due_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    due_model.fit(X_train, y_train)
    test_acc = due_model.score(X_test, y_test)

    active = model_df[model_df["return_date"].isna()].copy()
    if active.empty:
        return active, test_acc

    active_probs = due_model.predict_proba(active[features])[:, 1]
    active["late_probability"] = active_probs
    active["predicted_late"] = active["late_probability"] >= 0.5

    return active.sort_values("late_probability", ascending=False), test_acc


def build_dashboard(
    transactions: pd.DataFrame,
    demand_test: pd.DataFrame,
    availability_test: pd.DataFrame,
    lost_books: pd.DataFrame,
    due_predictions: pd.DataFrame,
    show_plot: bool,
) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(18, 15))

    # 1) Demand prediction graph: predicted vs actual monthly demand.
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

    # 2) Most popular books by total borrow count.
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

    # 3) Peak usage time graph using synthetic checkout-hour distribution.
    hourly = transactions.groupby("checkout_hour").size().reindex(range(8, 21), fill_value=0)
    ax = axes[1, 0]
    ax.plot(hourly.index, hourly.values, marker="o", color="#e76f51")
    ax.set_title("Peak Usage Time (Hourly Checkouts)")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Checkouts")

    # 4) Availability prediction comparison across all required models.
    ax = axes[1, 1]
    ax.plot(availability_test["date"], availability_test["available_books"], label="Actual", linewidth=2)
    ax.plot(availability_test["date"], availability_test["pred_linear"], label="Linear Regression", linestyle="--")
    ax.plot(
        availability_test["date"],
        availability_test["pred_holt_winters"],
        label="Holt-Winters",
        linestyle="--",
    )
    ax.plot(availability_test["date"], availability_test["pred_tree"], label="Decision Tree", linestyle="--")
    ax.set_title("Availability Prediction Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Available Books")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    # 5) Lost book detection summary by category.
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

    # 6) Due date violation prediction for currently borrowed books.
    ax = axes[2, 1]
    if due_predictions.empty:
        ax.text(0.5, 0.5, "No active borrows for due-date prediction", ha="center", va="center", fontsize=12)
        ax.axis("off")
    else:
        show = due_predictions.head(10).copy()
        show["label"] = show["book_title"].str.slice(0, 22) + " | " + show["user_role"].str.slice(0, 1).str.upper()
        show = show.sort_values("late_probability")
        ax.barh(show["label"], show["late_probability"], color="#6a4c93")
        ax.set_title("Predicted Due-Date Violation Risk (Active Borrows)")
        ax.set_xlabel("Predicted Late Probability")
        ax.set_ylabel("Book | Role")

    fig.suptitle("RFID Library Management Analytics Dashboard", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig("rfid_dashboard.png", dpi=150, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main(show_plot: bool = False) -> None:
    transactions, book_df = generate_synthetic_transactions(total_rows=1000)

    # Keep export in requested schema while preserving model helper columns in memory.
    export_cols = [
        "user_id",
        "book_id",
        "borrow_date",
        "return_date",
        "reservation_status",
        "overdue_status",
        "user_role",
        "book_category",
    ]
    export_df = transactions[export_cols].copy()

    export_df["borrow_date"] = export_df["borrow_date"].dt.strftime("%d-%m-%Y")
    export_df["return_date"] = export_df["return_date"].dt.strftime("%d-%m-%Y")
    export_df["return_date"] = export_df["return_date"].fillna("")
    export_df["reservation_status"] = export_df["reservation_status"].map({True: "TRUE", False: "FALSE"})

    export_df.to_csv("rfid_transactions_1000.csv", index=False)

    demand_test, demand_metrics = train_demand_model(transactions)
    availability_test, availability_metrics = train_availability_models(
        prepare_availability_series(transactions, book_df)
    )
    lost_books = detect_lost_books(transactions, overdue_threshold_days=14)
    due_predictions, due_model_accuracy = predict_due_date_violations(transactions)

    print("\n=== MODEL METRICS (MAE, R2) ===")
    print(f"{demand_metrics.name}: MAE={demand_metrics.mae:.3f}, R2={demand_metrics.r2:.3f}")
    for metric in availability_metrics.values():
        print(f"{metric.name}: MAE={metric.mae:.3f}, R2={metric.r2:.3f}")

    print("\n=== DUE DATE VIOLATION CLASSIFIER ===")
    print(f"Validation Accuracy: {due_model_accuracy:.3f}")

    print("\n=== LOST BOOK DETECTION ===")
    print(f"Lost candidates: {len(lost_books)}")
    if not lost_books.empty:
        print(
            lost_books[["user_id", "book_title", "borrow_date", "borrow_age", "book_category"]]
            .head(10)
            .to_string(index=False)
        )

    print("\n=== TOP PREDICTED LATE ACTIVE BORROWS ===")
    if due_predictions.empty:
        print("No active borrows available for due-date violation prediction.")
    else:
        print(
            due_predictions[
                ["user_id", "book_title", "borrow_date", "late_probability", "predicted_late"]
            ]
            .head(10)
            .to_string(index=False)
        )

    build_dashboard(
        transactions,
        demand_test,
        availability_test,
        lost_books,
        due_predictions,
        show_plot=show_plot,
    )
    print("\nDashboard saved to: rfid_dashboard.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFID Library Management Analytics")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display dashboard window in addition to saving image.",
    )
    args = parser.parse_args()
    main(show_plot=args.show)
