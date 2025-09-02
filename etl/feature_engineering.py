import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from datetime import datetime

# Load env vars
load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Create engine with error handling
try:
    engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("✅ Database connection successful")
except Exception as e:
    print(f"❌ Database connection failed: {e}")
    raise

def build_features():
    try:
        # ----------------
        # Extract raw data with error handling
        # ----------------
        print("📥 Extracting data from database...")
        
        # CHANGED: Removed _clean suffix from table names
        tickets = pd.read_sql("SELECT * FROM tickets", engine)
        events = pd.read_sql("SELECT * FROM ticket_events", engine)
        customers = pd.read_sql("SELECT * FROM customers", engine)
        agents = pd.read_sql("SELECT * FROM agents", engine)
        
        # Check if tables are empty
        if tickets.empty or events.empty or customers.empty or agents.empty:
            raise ValueError("One or more database tables are empty")
        
        print(f"✅ Loaded data: {len(tickets)} tickets, {len(events)} events, {len(customers)} customers, {len(agents)} agents")

        # ----------------
        # Feature: Ticket age (in hours)
        # ----------------
        # Convert to datetime if not already
        tickets["created_at"] = pd.to_datetime(tickets["created_at"])
        current_time = pd.Timestamp.now()
        tickets["ticket_age_hrs"] = (current_time - tickets["created_at"]).dt.total_seconds() / 3600

        # ----------------
        # Feature: Number of transfers
        # ----------------
        # CHANGED: Using "Updated" event type instead of "transfer" since your data has "Created", "Updated", "Commented", "Closed"
        transfers = (
            events[events["event_type"] == "Updated"]
            .groupby("ticket_id")
            .size()
            .reset_index(name="num_transfers")
        )

        # ----------------
        # Feature: Messages in first 2 hours
        # ----------------
        # Ensure datetime conversion
        events["event_time"] = pd.to_datetime(events["event_time"])
        events = events.merge(tickets[["ticket_id", "created_at"]], on="ticket_id", how="left")
        
        # CHANGED: Using "Commented" event type instead of "message"
        first_2h = events[
            (events["event_type"] == "Commented") & 
            (events["event_time"] <= events["created_at"] + pd.Timedelta(hours=2))
        ]
        msg_2h = first_2h.groupby("ticket_id").size().reset_index(name="num_msgs_first_2h")

        # ----------------
        # Feature: Avg response time (simplified proxy)
        # ----------------
        # CHANGED: Using "Commented" event type instead of "message"
        message_events = events[events["event_type"] == "Commented"].copy()
        
        response_time = (
            message_events
            .groupby("ticket_id")
            .event_time
            .apply(lambda x: (x.max() - x.min()).total_seconds() / max(len(x) - 1, 1) if len(x) > 1 else -1)
            .reset_index(name="avg_response_time_secs")
        )

        # ----------------
        # Join All Features
        # ----------------
        features = tickets.merge(customers[["customer_id", "region"]], on="customer_id", how="left")  # CHANGED: using region instead of tier
        features = features.merge(agents[["agent_id", "experience_years"]], on="agent_id", how="left")
        features = features.merge(transfers, on="ticket_id", how="left")
        features = features.merge(msg_2h, on="ticket_id", how="left")
        features = features.merge(response_time, on="ticket_id", how="left")

        # Fill nulls
        features = features.fillna({
            "num_transfers": 0,
            "num_msgs_first_2h": 0,
            "avg_response_time_secs": -1,  # -1 means no agent response yet
            "region": "unknown",  # CHANGED: using region instead of tier
            "experience_years": 0
        })

        # Keep only needed columns
        features_core = features[[
            "ticket_id", "ticket_age_hrs", "num_transfers", "num_msgs_first_2h",
            "avg_response_time_secs", "region", "experience_years"  # CHANGED: using region instead of tier
        ]]

        # ----------------
        # Load into DB
        # ----------------
        print("💾 Saving features to database...")
        features_core.to_sql("features_core", engine, if_exists="replace", index=False)
        print(f"✅ Created features_core with {len(features_core)} rows")
        
        # Show sample of the features
        print("\n📊 Sample of generated features:")
        print(features_core.head())
        
        return features_core
        
    except Exception as e:
        print(f"❌ Error in build_features: {e}")
        raise

if __name__ == "__main__":
    build_features()