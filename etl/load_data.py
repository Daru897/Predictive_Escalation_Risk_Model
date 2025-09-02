import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
import os

# Load environment variables
load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Create SQLAlchemy engine
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")


# ----------------------------
# Generic Cleaning Utilities
# ----------------------------
def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply common cleaning rules to all tables"""
    df = df.dropna(how="all")  # drop fully empty rows
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


# ----------------------------
# Table-Specific Cleaning
# ----------------------------
def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_clean(df)
    # Ensure emails are lowercase + remove obvious invalids
    if "email" in df.columns:
        df["email"] = df["email"].str.lower()
        df = df[df["email"].str.contains("@", na=False)]
    # Normalize region values (title case)
    if "region" in df.columns:
        df["region"] = df["region"].str.title()
    return df


def clean_agents(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_clean(df)
    # Ensure experience_years is numeric and non-negative
    if "experience_years" in df.columns:
        df["experience_years"] = pd.to_numeric(df["experience_years"], errors="coerce").fillna(0)
        df = df[df["experience_years"] >= 0]
    # Normalize department names
    if "department" in df.columns:
        df["department"] = df["department"].str.title()
    return df


def clean_tickets(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_clean(df)
    # Parse created_at
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df = df.dropna(subset=["created_at"])
    # Normalize status
    if "status" in df.columns:
        df["status"] = df["status"].str.lower().replace({
            "open ": "open", "closed ": "closed"
        })
        df = df[df["status"].isin(["open", "closed", "pending", "resolved"])]
    # Normalize priority
    if "priority" in df.columns:
        df["priority"] = df["priority"].str.title()
    return df


def clean_ticket_events(df: pd.DataFrame) -> pd.DataFrame:
    df = basic_clean(df)
    # Parse event_time
    if "event_time" in df.columns:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
        df = df.dropna(subset=["event_time"])
    # Normalize event_type
    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].str.lower().replace({
            "transfered": "transfer", "xfer": "transfer"
        })
        valid_events = ["message", "transfer", "close", "escalate"]
        df = df[df["event_type"].isin(valid_events)]
    # Clean event_description (remove PII)
    if "event_description" in df.columns:
        df["event_description"] = (
            df["event_description"]
            .str.replace(r"\b\d{10}\b", "[PHONE]", regex=True)
            .str.replace(r"\S+@\S+", "[EMAIL]", regex=True)
        )
    return df


# ----------------------------
# ETL Runner
# ----------------------------
def etl_table(source_table, target_table, cleaner_func):
    # Extract
    df = pd.read_sql(f"SELECT * FROM {source_table}", engine)
    rows_before = len(df)

    # Transform
    df_clean = cleaner_func(df)
    rows_after = len(df_clean)

    # Load (replace existing clean table each run)
    df_clean.to_sql(target_table, engine, if_exists="replace", index=False)

    # Log
    with engine.connect() as conn:
        conn.execute(
            text("INSERT INTO etl_logs (table_name, rows_inserted, run_time) VALUES (:table, :rows, NOW())"),
            {"table": target_table, "rows": rows_after}
        )

    print(f"[{datetime.now()}] {source_table} → {target_table} | Raw: {rows_before} | Clean: {rows_after}")


if __name__ == "__main__":
    etl_table("customers", "customers_clean", clean_customers)
    etl_table("agents", "agents_clean", clean_agents)
    etl_table("tickets", "tickets_clean", clean_tickets)
    etl_table("ticket_events", "ticket_events_clean", clean_ticket_events)
