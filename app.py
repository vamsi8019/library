from datetime import datetime
import hashlib
import uuid

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

try:
    import serial
except Exception:
    serial = None

from rfid_library_management import (
    TODAY,
    detect_lost_books,
    predict_due_date_violations,
    prepare_availability_series,
    train_availability_models,
    train_demand_model,
)


st.set_page_config(
    page_title="RFID AI Smart Library",
    page_icon="RFID",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Manrope:wght@400;500;600&display=swap');

        .stApp {
            font-family: 'Manrope', sans-serif;
            background:
                radial-gradient(1200px 500px at 0% -10%, rgba(18, 140, 126, 0.17), transparent),
                radial-gradient(900px 450px at 100% 0%, rgba(242, 141, 51, 0.16), transparent),
                linear-gradient(180deg, #f5f8f9 0%, #edf3f6 100%);
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 1.6rem;
        }

        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .hero {
            background: linear-gradient(135deg, #102a43 0%, #164e63 48%, #1f7a8c 100%);
            color: #f5fbff;
            border-radius: 18px;
            padding: 1rem 1.2rem;
            box-shadow: 0 12px 26px rgba(10, 30, 60, 0.25);
            margin-bottom: 0.8rem;
            animation: riseIn 0.55s ease both;
        }

        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.15rem;
        }

        .hero-sub {
            font-size: 1rem;
            opacity: 0.93;
        }

        .glass {
            background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(246,250,252,0.86));
            border: 1px solid rgba(20, 60, 90, 0.12);
            border-radius: 16px;
            padding: 0.9rem 1rem;
            box-shadow: 0 10px 24px rgba(18, 36, 58, 0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            animation: riseIn 0.5s ease both;
        }

        .glass:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 28px rgba(18, 36, 58, 0.11);
        }

        .panel {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(20, 60, 90, 0.12);
            border-radius: 18px;
            padding: 1rem 1rem 0.9rem;
            box-shadow: 0 12px 26px rgba(18, 36, 58, 0.08);
            margin-top: 0.75rem;
            animation: riseIn 0.55s ease both;
        }

        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.15rem;
            font-weight: 700;
            color: #0e2a38;
            margin: 0 0 0.15rem 0;
        }

        .section-subtitle {
            color: #577083;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }

        .badge-row {
            display: flex;
            gap: 0.45rem;
            flex-wrap: wrap;
            margin: 0.55rem 0 0.15rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.28rem 0.68rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.14);
            color: #eef7fb;
            border: 1px solid rgba(255, 255, 255, 0.22);
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .metric-chip {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            background: rgba(14, 42, 56, 0.05);
            color: #274455;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 0.8rem;
        }

        .sidebar-card {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(20, 60, 90, 0.12);
            border-radius: 18px;
            padding: 0.9rem;
            box-shadow: 0 10px 20px rgba(18, 36, 58, 0.06);
        }

        .sidebar-card h3 {
            margin: 0 0 0.6rem 0;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1rem;
        }

        div[data-testid="stRadio"] label {
            background: rgba(20, 60, 90, 0.04);
            border: 1px solid rgba(20, 60, 90, 0.11);
            border-radius: 999px;
            padding: 0.25rem 0.5rem;
            transition: border-color 0.18s ease, background 0.18s ease;
        }

        div[data-testid="stRadio"] label:has(input:checked) {
            border-color: rgba(18, 140, 126, 0.65);
            background: rgba(18, 140, 126, 0.08);
        }

        .sidebar-card .stButton > button {
            border-radius: 999px;
        }

        .stButton > button {
            border-radius: 999px;
            border: 1px solid rgba(20, 60, 90, 0.14);
            background: linear-gradient(135deg, #ffffff, #eef6fa);
            color: #153447;
            box-shadow: 0 8px 18px rgba(18, 36, 58, 0.05);
        }

        .stButton > button:hover {
            border-color: rgba(18, 140, 126, 0.45);
            transform: translateY(-1px);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(20, 60, 90, 0.12);
            box-shadow: 0 10px 24px rgba(18, 36, 58, 0.06);
        }

        .kpi-label {
            font-size: 0.85rem;
            color: #466173;
        }

        .kpi-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.5rem;
            font-weight: 700;
            color: #0e2a38;
        }

        @media (max-width: 768px) {
            .hero {
                padding: 0.9rem 1rem;
                border-radius: 16px;
            }

            .hero-title {
                font-size: 1.45rem;
                line-height: 1.15;
            }

            .hero-sub {
                font-size: 0.92rem;
            }

            .glass {
                padding: 0.8rem 0.85rem;
                border-radius: 14px;
            }

            .kpi-value {
                font-size: 1.25rem;
            }

            section[data-testid="stSidebar"] {
                min-width: 100% !important;
                width: 100% !important;
            }

            section[data-testid="stSidebar"] > div {
                width: 100% !important;
            }

            div[data-testid="stHorizontalBlock"] {
                flex-direction: column;
                gap: 0.75rem;
            }

            div[data-testid="column"] {
                width: 100% !important;
                flex: 1 1 100% !important;
            }

            .stButton > button,
            .stTextInput input,
            .stSelectbox div,
            .stSlider,
            .stMultiSelect div {
                width: 100%;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def rfid_token(prefix: str, raw_value: str) -> str:
    token = hashlib.md5(raw_value.encode("utf-8")).hexdigest()[:10].upper()
    return f"{prefix}-{token}"


def role_due_days(role: str) -> int:
    lookup = {"student": 14, "faculty": 21, "staff": 10}
    return lookup.get(str(role).lower(), 14)


def load_seed_history() -> pd.DataFrame:
    try:
        base = pd.read_csv("rfid_transactions_1000.csv")
        base["borrow_date"] = pd.to_datetime(base["borrow_date"], format="%d-%m-%Y", errors="coerce")
        base["return_date"] = pd.to_datetime(base["return_date"], format="%d-%m-%Y", errors="coerce")
        base["reservation_status"] = (
            base["reservation_status"].astype(str).str.strip().str.upper().map({"TRUE": True, "FALSE": False})
        )
        base["reservation_status"] = base["reservation_status"].fillna(False)
        base["overdue_status"] = pd.to_numeric(base["overdue_status"], errors="coerce").fillna(0).astype(int)
        base = base.dropna(subset=["borrow_date"]).copy()
    except Exception:
        base = pd.DataFrame(
            columns=[
                "user_id",
                "book_id",
                "borrow_date",
                "return_date",
                "reservation_status",
                "overdue_status",
                "user_role",
                "book_category",
            ]
        )

    if base.empty:
        return base

    base["user_rfid"] = base["user_id"].apply(lambda v: rfid_token("USER", str(v)))
    base["book_rfid"] = base["book_id"].apply(lambda v: rfid_token("BOOK", str(v)))
    base["book_title"] = base["book_id"].apply(lambda v: f"Book-{str(v)[:8]}")
    base["checkout_hour"] = np.random.randint(8, 21, size=len(base))
    base["loan_id"] = [str(uuid.uuid4()) for _ in range(len(base))]
    return base


def build_catalogs(loan_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = (
        loan_df[["user_id", "user_rfid", "user_role"]]
        .drop_duplicates()
        .copy()
    )
    if users.empty:
        users = pd.DataFrame(columns=["user_id", "user_rfid", "user_role", "name", "status"])
    users["name"] = users["user_id"].astype(str).str[:8].apply(lambda x: f"Member-{x}")
    users["status"] = "active"

    books = (
        loan_df[["book_id", "book_rfid", "book_title", "book_category"]]
        .drop_duplicates()
        .copy()
    )
    if books.empty:
        books = pd.DataFrame(columns=["book_id", "book_rfid", "book_title", "book_category", "total_copies"])
    books["total_copies"] = 1
    books["status"] = "available"
    books["holder_user_rfid"] = ""
    books["due_date"] = pd.NaT

    open_loans = loan_df[loan_df["return_date"].isna()].copy()
    if not open_loans.empty and not books.empty:
        latest_open = open_loans.sort_values("borrow_date").groupby("book_rfid").tail(1)
        book_map = latest_open.set_index("book_rfid")
        hit = books["book_rfid"].isin(book_map.index)
        books.loc[hit, "status"] = "checked_out"
        books.loc[hit, "holder_user_rfid"] = books.loc[hit, "book_rfid"].map(book_map["user_rfid"])

        due_dates = []
        for _, row in books.loc[hit, ["book_rfid"]].iterrows():
            b_rfid = row["book_rfid"]
            record = book_map.loc[b_rfid]
            due = record["borrow_date"] + pd.Timedelta(days=role_due_days(record["user_role"]))
            due_dates.append(due)
        books.loc[hit, "due_date"] = due_dates

    return users.reset_index(drop=True), books.reset_index(drop=True)


def init_state() -> None:
    if "loan_df" in st.session_state:
        return

    loan_df = load_seed_history()
    users_df, books_df = build_catalogs(loan_df)

    st.session_state.loan_df = loan_df
    st.session_state.users_df = users_df
    st.session_state.books_df = books_df
    st.session_state.scan_log = []
    st.session_state.scanned_user_rfid = ""
    st.session_state.scanned_book_rfid = ""
    st.session_state.pending_raw_uid = ""
    st.session_state.uid_map = {}


def rebuild_catalogs_from_loans() -> None:
    users_df, books_df = build_catalogs(st.session_state.loan_df)

    if not st.session_state.users_df.empty:
        extra_users = st.session_state.users_df[~st.session_state.users_df["user_rfid"].isin(users_df["user_rfid"])]
        users_df = pd.concat([users_df, extra_users], ignore_index=True)

    if not st.session_state.books_df.empty:
        extra_books = st.session_state.books_df[~st.session_state.books_df["book_rfid"].isin(books_df["book_rfid"])]
        books_df = pd.concat([books_df, extra_books], ignore_index=True)

    st.session_state.users_df = users_df.drop_duplicates(subset=["user_rfid"]).reset_index(drop=True)
    st.session_state.books_df = books_df.drop_duplicates(subset=["book_rfid"]).reset_index(drop=True)


def log_scan(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.scan_log.insert(0, f"[{stamp}] {message}")
    st.session_state.scan_log = st.session_state.scan_log[:20]


def read_scans_from_serial(port: str, baud_rate: int, max_lines: int = 5) -> list[str]:
    """Read RFID tags from a serial-connected reader (one tag per line)."""
    if serial is None:
        return []

    tags: list[str] = []
    try:
        with serial.Serial(port=port, baudrate=baud_rate, timeout=0.25) as reader:
            for _ in range(max_lines):
                raw = reader.readline()
                if not raw:
                    break
                tag = raw.decode("utf-8", errors="ignore").strip()
                if tag:
                    tags.append(tag)
    except Exception as exc:
        log_scan(f"Serial read error on {port}: {exc}")
    return tags


def scan_tag(raw_scan: str) -> None:
    tag = str(raw_scan).strip().upper()
    if not tag:
        return

    if tag in st.session_state.uid_map:
        mapped = st.session_state.uid_map[tag]
        log_scan(f"Mapped raw UID {tag} -> {mapped}")
        tag = mapped

    if tag.startswith("USER-"):
        st.session_state.scanned_user_rfid = tag
        log_scan(f"User tag scanned: {tag}")
    elif tag.startswith("BOOK-"):
        st.session_state.scanned_book_rfid = tag
        log_scan(f"Book tag scanned: {tag}")
    else:
        st.session_state.pending_raw_uid = tag
        log_scan(f"Raw UID captured: {tag}. Assign as USER or BOOK to continue.")


def assign_pending_uid(target: str) -> str:
    raw_uid = st.session_state.pending_raw_uid
    if not raw_uid:
        return "No pending raw UID to assign."

    if target not in {"USER", "BOOK"}:
        return "Invalid assignment target."

    mapped = f"{target}-{hashlib.md5(raw_uid.encode('utf-8')).hexdigest()[:10].upper()}"
    st.session_state.uid_map[raw_uid] = mapped
    st.session_state.pending_raw_uid = ""
    scan_tag(mapped)
    return f"Mapped {raw_uid} as {mapped}"


def register_scanned_user() -> str:
    user_rfid = st.session_state.scanned_user_rfid
    if not user_rfid:
        return "No user RFID scanned."

    users_df = st.session_state.users_df
    if user_rfid in set(users_df["user_rfid"]):
        return f"User already exists for {user_rfid}."

    user_id = str(uuid.uuid4())
    new_user = pd.DataFrame(
        [
            {
                "user_id": user_id,
                "user_rfid": user_rfid,
                "user_role": "student",
                "name": f"Member-{user_rfid[-6:]}",
                "status": "active",
            }
        ]
    )
    st.session_state.users_df = pd.concat([users_df, new_user], ignore_index=True)
    return f"New user registered from RFID: {user_rfid}"


def register_scanned_book() -> str:
    book_rfid = st.session_state.scanned_book_rfid
    if not book_rfid:
        return "No book RFID scanned."

    books_df = st.session_state.books_df
    if book_rfid in set(books_df["book_rfid"]):
        return f"Book already exists for {book_rfid}."

    new_book = pd.DataFrame(
        [
            {
                "book_id": str(uuid.uuid4()),
                "book_rfid": book_rfid,
                "book_title": f"Book-{book_rfid[-6:]}",
                "book_category": "fiction",
                "total_copies": 1,
                "status": "available",
                "holder_user_rfid": "",
                "due_date": pd.NaT,
            }
        ]
    )
    st.session_state.books_df = pd.concat([books_df, new_book], ignore_index=True)
    return f"New book registered from RFID: {book_rfid}"


def checkout_by_scan() -> str:
    user_rfid = st.session_state.scanned_user_rfid
    book_rfid = st.session_state.scanned_book_rfid

    if not user_rfid or not book_rfid:
        return "Scan both USER RFID and BOOK RFID before checkout."

    users_df = st.session_state.users_df
    books_df = st.session_state.books_df

    if user_rfid not in set(users_df["user_rfid"]):
        return "User RFID not registered. Register scanned user first."
    if book_rfid not in set(books_df["book_rfid"]):
        return "Book RFID not registered. Register scanned book first."

    book_row = books_df[books_df["book_rfid"] == book_rfid].iloc[0]
    if str(book_row["status"]).lower() == "checked_out":
        return "Book is already checked out."

    user_row = users_df[users_df["user_rfid"] == user_rfid].iloc[0]

    new_loan = pd.DataFrame(
        [
            {
                "loan_id": str(uuid.uuid4()),
                "user_id": user_row["user_id"],
                "book_id": book_row["book_id"],
                "user_rfid": user_rfid,
                "book_rfid": book_rfid,
                "borrow_date": pd.Timestamp(datetime.now().date()),
                "return_date": pd.NaT,
                "reservation_status": False,
                "overdue_status": 0,
                "user_role": user_row["user_role"],
                "book_category": book_row["book_category"],
                "book_title": book_row["book_title"],
                "checkout_hour": datetime.now().hour,
            }
        ]
    )

    st.session_state.loan_df = pd.concat([st.session_state.loan_df, new_loan], ignore_index=True)
    rebuild_catalogs_from_loans()
    return f"Checkout completed. {book_rfid} -> {user_rfid}"


def return_by_scan() -> str:
    user_rfid = st.session_state.scanned_user_rfid
    book_rfid = st.session_state.scanned_book_rfid

    if not book_rfid:
        return "Scan BOOK RFID before return."

    loan_df = st.session_state.loan_df.copy()
    open_mask = (loan_df["book_rfid"] == book_rfid) & (loan_df["return_date"].isna())

    if user_rfid:
        open_mask = open_mask & (loan_df["user_rfid"] == user_rfid)

    idx = loan_df[open_mask].index
    if len(idx) == 0:
        return "No active open loan found for scanned RFID tags."

    target_idx = idx[-1]
    return_date = pd.Timestamp(datetime.now().date())
    borrow_date = pd.to_datetime(loan_df.loc[target_idx, "borrow_date"])
    role = str(loan_df.loc[target_idx, "user_role"])
    overdue = int(return_date > borrow_date + pd.Timedelta(days=role_due_days(role)))

    loan_df.loc[target_idx, "return_date"] = return_date
    loan_df.loc[target_idx, "overdue_status"] = overdue

    st.session_state.loan_df = loan_df
    rebuild_catalogs_from_loans()
    return f"Return completed for {book_rfid}."


def reserve_by_scan() -> str:
    user_rfid = st.session_state.scanned_user_rfid
    book_rfid = st.session_state.scanned_book_rfid

    if not user_rfid or not book_rfid:
        return "Scan both USER RFID and BOOK RFID before reservation."

    users_df = st.session_state.users_df
    books_df = st.session_state.books_df

    if user_rfid not in set(users_df["user_rfid"]):
        return "User RFID not registered."
    if book_rfid not in set(books_df["book_rfid"]):
        return "Book RFID not registered."

    user_row = users_df[users_df["user_rfid"] == user_rfid].iloc[0]
    book_row = books_df[books_df["book_rfid"] == book_rfid].iloc[0]

    new_res = pd.DataFrame(
        [
            {
                "loan_id": str(uuid.uuid4()),
                "user_id": user_row["user_id"],
                "book_id": book_row["book_id"],
                "user_rfid": user_rfid,
                "book_rfid": book_rfid,
                "borrow_date": pd.Timestamp(datetime.now().date()),
                "return_date": pd.NaT,
                "reservation_status": True,
                "overdue_status": 0,
                "user_role": user_row["user_role"],
                "book_category": book_row["book_category"],
                "book_title": book_row["book_title"],
                "checkout_hour": datetime.now().hour,
            }
        ]
    )

    st.session_state.loan_df = pd.concat([st.session_state.loan_df, new_res], ignore_index=True)
    rebuild_catalogs_from_loans()
    return f"Reservation captured for {book_rfid} by {user_rfid}."


def hero_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">RFID Based Smart Library Management System</div>
            <div class="hero-sub">AI powered operations using RFID-only scan flow for user and book management.</div>
            <div class="badge-row">
                <span class="badge">RFID-first workflow</span>
                <span class="badge">AI analytics</span>
                <span class="badge">Live updates</span>
                <span class="badge">Responsive layout</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="glass">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def metric_chip(text: str) -> None:
    st.markdown(f'<div class="metric-chip">{text}</div>', unsafe_allow_html=True)


def dashboard_page() -> None:
    loan_df = st.session_state.loan_df
    books_df = st.session_state.books_df
    users_df = st.session_state.users_df

    section_header("Dashboard Overview", "At-a-glance activity, usage, and catalog health.")
    metric_chip(f"Last sync: {datetime.now().strftime('%H:%M')}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Members", f"{len(users_df):,}")
    with c2:
        kpi_card("Total Tagged Books", f"{len(books_df):,}")
    with c3:
        active_loans = int(loan_df["return_date"].isna().sum())
        kpi_card("Active Loans", f"{active_loans:,}")
    with c4:
        available_books = int((books_df["status"] == "available").sum())
        kpi_card("Available Books", f"{available_books:,}")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    col1, col2 = st.columns([1.7, 1])
    with col1:
        monthly = loan_df.set_index("borrow_date").resample("MS").size().reset_index(name="count")
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(monthly["borrow_date"], monthly["count"], color="#0077b6", marker="o", linewidth=2.2)
        ax.fill_between(monthly["borrow_date"], monthly["count"], color="#cbe9f6", alpha=0.6)
        ax.set_title("Borrow Trend")
        ax.set_xlabel("Month")
        ax.set_ylabel("Borrows")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.18)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col2:
        by_role = loan_df.groupby("user_role").size().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(by_role.values, labels=by_role.index, autopct="%1.1f%%", startangle=130, wedgeprops={"linewidth": 1, "edgecolor": "white"})
        ax.set_title("Borrow Mix by Role")
        st.pyplot(fig, width="stretch")
        plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)


def rfid_ops_page() -> None:
    section_header("RFID Scan Console", "Operations are executed only by scanned RFID tags.")

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        scan_input = st.text_input(
            "Scan RFID Tag",
            value="",
            placeholder="Example: USER-AB12CD34EF or BOOK-9F8E7D6C5B",
        )
        if st.button("Capture Scan", type="primary"):
            scan_tag(scan_input)

    with c2:
        st.markdown("**Current Scan Buffer**")
        st.write(f"User RFID: {st.session_state.scanned_user_rfid or 'None'}")
        st.write(f"Book RFID: {st.session_state.scanned_book_rfid or 'None'}")
        st.write(f"Pending Raw UID: {st.session_state.pending_raw_uid or 'None'}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    raw1, raw2 = st.columns(2)
    with raw1:
        if st.button("Assign Pending UID as User"):
            st.success(assign_pending_uid("USER"))
    with raw2:
        if st.button("Assign Pending UID as Book"):
            st.success(assign_pending_uid("BOOK"))

    a1, a2, a3, a4, a5 = st.columns(5)
    with a1:
        if st.button("Register User Card"):
            st.success(register_scanned_user())
    with a2:
        if st.button("Register Book Tag"):
            st.success(register_scanned_book())
    with a3:
        if st.button("Checkout"):
            st.info(checkout_by_scan())
    with a4:
        if st.button("Return"):
            st.info(return_by_scan())
    with a5:
        if st.button("Reserve"):
            st.info(reserve_by_scan())
    st.markdown("</div>", unsafe_allow_html=True)

    section_header("Recent RFID Scan Log", "Most recent scan events and mappings.")
    if not st.session_state.scan_log:
        st.write("No scans yet.")
    else:
        for item in st.session_state.scan_log:
            st.write(item)

    if st.session_state.uid_map:
        st.markdown("### Raw UID Mapping Table")
        map_df = pd.DataFrame(
            [
                {"raw_uid": raw_uid, "mapped_rfid": mapped}
                for raw_uid, mapped in st.session_state.uid_map.items()
            ]
        )
        st.dataframe(map_df, width="stretch", hide_index=True)


def registry_page() -> None:
    section_header("RFID Registry", "Users, books, and recent loans live in one place.")

    users_df = st.session_state.users_df.copy()
    books_df = st.session_state.books_df.copy()

    tabs = st.tabs(["Users by RFID", "Books by RFID", "Recent Loans"])

    with tabs[0]:
        st.dataframe(
            users_df[["user_rfid", "name", "user_role", "status"]].sort_values("name"),
            width="stretch",
            hide_index=True,
        )

    with tabs[1]:
        show = books_df[["book_rfid", "book_title", "book_category", "status", "holder_user_rfid", "due_date"]].copy()
        show["due_date"] = pd.to_datetime(show["due_date"], errors="coerce").dt.strftime("%Y-%m-%d")
        show["due_date"] = show["due_date"].fillna("")
        st.dataframe(show.sort_values("book_title"), width="stretch", hide_index=True)

    with tabs[2]:
        loan_df = st.session_state.loan_df.copy().sort_values("borrow_date", ascending=False)
        view = loan_df[
            [
                "borrow_date",
                "return_date",
                "user_rfid",
                "book_rfid",
                "book_title",
                "reservation_status",
                "overdue_status",
            ]
        ].head(40)
        st.dataframe(view, width="stretch", hide_index=True)


def ai_page() -> None:
    section_header("AI Powered Insights", "Model quality, demand prediction, and exception detection.")

    loan_df = st.session_state.loan_df.copy()
    books_df = st.session_state.books_df.copy()

    if loan_df.empty or len(loan_df) < 30:
        st.warning("Not enough loan history to train models. Use RFID checkout operations to generate more history.")
        return

    model_df = loan_df.copy()
    model_df["borrow_date"] = pd.to_datetime(model_df["borrow_date"], errors="coerce")
    model_df["return_date"] = pd.to_datetime(model_df["return_date"], errors="coerce")
    model_df = model_df.dropna(subset=["borrow_date"]).copy()

    inventory_df = books_df[["book_id", "book_title", "book_category", "total_copies"]].drop_duplicates()

    demand_test, demand_metrics = train_demand_model(model_df)
    availability_test, availability_metrics = train_availability_models(
        prepare_availability_series(model_df, inventory_df)
    )
    lost_books = detect_lost_books(model_df, overdue_threshold_days=14)
    due_predictions, due_acc = predict_due_date_violations(model_df)

    metrics = pd.DataFrame(
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
        }
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.dataframe(metrics, width="stretch", hide_index=True)
    st.info(f"Due Date Violation Prediction Accuracy: {due_acc:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(demand_test["borrow_date"], demand_test["actual_demand"], marker="o", label="Actual")
    ax.plot(demand_test["borrow_date"], demand_test["predicted_demand"], marker="o", linestyle="--", label="Predicted")
    ax.set_title("Book Demand Prediction")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    top_books = (
        model_df.groupby("book_title")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .sort_values(ascending=True)
    )
    ax = axes[0, 1]
    ax.barh(top_books.index, top_books.values, color="#ff7f11")
    ax.set_title("Most Popular Books")

    ax = axes[1, 0]
    ax.plot(availability_test["date"], availability_test["available_books"], label="Actual", linewidth=2)
    ax.plot(availability_test["date"], availability_test["pred_linear"], linestyle="--", label="Linear")
    ax.plot(availability_test["date"], availability_test["pred_holt_winters"], linestyle="--", label="Holt-Winters")
    ax.plot(availability_test["date"], availability_test["pred_tree"], linestyle="--", label="Tree")
    ax.set_title("Availability Prediction Comparison")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    hourly = model_df.groupby("checkout_hour").size().reindex(range(8, 21), fill_value=0)
    ax = axes[1, 1]
    ax.plot(hourly.index, hourly.values, marker="o", color="#0081a7")
    ax.set_title("Peak Usage Time")

    fig.tight_layout()
    st.pyplot(fig, width="stretch")
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    section_header("Lost Book Detection", "Books flagged by overdue and missing-return heuristics.")
    if lost_books.empty:
        st.success("No lost books flagged.")
    else:
        st.dataframe(
            lost_books[["user_rfid", "book_rfid", "book_title", "borrow_date", "book_category"]].head(25),
            width="stretch",
            hide_index=True,
        )

    section_header("Due Date Violation Prediction", "Active loans scored for late-return risk.")
    if due_predictions.empty:
        st.warning("No active borrows to score.")
    else:
        st.dataframe(
            due_predictions[["user_rfid", "book_rfid", "book_title", "late_probability", "predicted_late"]].head(25),
            width="stretch",
            hide_index=True,
        )


def live_simulation_tick() -> None:
    if st.session_state.users_df.empty or st.session_state.books_df.empty:
        return

    users = st.session_state.users_df
    books = st.session_state.books_df

    picked_user = users.sample(1).iloc[0]["user_rfid"]
    available_books = books[books["status"] == "available"]

    if available_books.empty:
        return

    picked_book = available_books.sample(1).iloc[0]["book_rfid"]
    st.session_state.scanned_user_rfid = picked_user
    st.session_state.scanned_book_rfid = picked_book
    msg = checkout_by_scan()
    log_scan(f"Auto simulated event: {msg}")


def main() -> None:
    apply_styles()
    init_state()

    page_options = {
        "Dashboard": "DASHBOARD  Activity Overview",
        "RFID Operations": "OPERATIONS  Scan and Process",
        "Registry": "REGISTRY  Users and Books",
        "AI Insights": "AI INSIGHTS  Predictions and Risk",
    }

    with st.sidebar:
        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.header("Application Navigation")
        selected = st.radio(
            "Select Screen",
            list(page_options.values()),
            index=0,
        )
        page = next((k for k, v in page_options.items() if v == selected), "Dashboard")

        st.markdown("---")
        auto_mode = st.toggle("Real-time simulation", value=False)
        if auto_mode:
            seconds = st.slider("Refresh interval (seconds)", 5, 60, 15)
            st_autorefresh(interval=seconds * 1000, key="rfid_live_tick")
            live_simulation_tick()

        if st.button("Clear scan buffer"):
            st.session_state.scanned_user_rfid = ""
            st.session_state.scanned_book_rfid = ""

        st.markdown("---")
        st.subheader("RFID Reader Input")
        serial_enabled = st.toggle("Enable Serial RFID Reader", value=False)
        serial_port = st.text_input("Serial Port", value="COM3")
        baud_rate = st.selectbox("Baud Rate", [9600, 19200, 38400, 57600, 115200], index=0)
        auto_map_target = st.selectbox("Auto-map unknown raw UID as", ["Disabled", "USER", "BOOK"], index=0)

        if serial_enabled:
            if serial is None:
                st.warning("pyserial not installed. Install requirements and restart app.")
            else:
                serial_tags = read_scans_from_serial(serial_port, baud_rate)
                for tag in serial_tags:
                    scan_tag(tag)
                    if st.session_state.pending_raw_uid and auto_map_target in {"USER", "BOOK"}:
                        msg = assign_pending_uid(auto_map_target)
                        log_scan(msg)
                if serial_tags:
                    st.success(f"Captured {len(serial_tags)} tag(s) from {serial_port}")
            st.markdown('</div>', unsafe_allow_html=True)

    hero_header()

    if page == "Dashboard":
        dashboard_page()
    elif page == "RFID Operations":
        rfid_ops_page()
    elif page == "Registry":
        registry_page()
    else:
        ai_page()

    st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | System date anchor: {TODAY.date()}")


if __name__ == "__main__":
    main()
