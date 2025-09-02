import pandas as pd
from sqlalchemy import create_engine, text as sql_text  # Renamed import to avoid conflict
from dotenv import load_dotenv
import os
import logging
from typing import Optional
import numpy as np
from datetime import datetime

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# DB Setup with optimized engine
# ----------------------------
load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Optimized engine with connection pooling
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    execution_options={"isolation_level": "AUTOCOMMIT"}
)

# ----------------------------
# Optimized Feature Store Building
# ----------------------------
def build_feature_store() -> Optional[pd.DataFrame]:
    """
    Build and optimize the feature store by merging core, text, and ticket features
    """
    try:
        logger.info("📥 Loading features from database...")
        
        # Load tables with optimized queries
        table_queries = {
            'core': "SELECT ticket_id, ticket_age_hrs, num_transfers, num_msgs_first_2h, avg_response_time_secs, region, experience_years FROM features_core",
            'text_features': "SELECT ticket_id, sentiment FROM features_text",  # Renamed variable
            'tickets': "SELECT ticket_id, status, priority, created_at FROM tickets"
        }
        
        # Load tables
        core = pd.read_sql(table_queries['core'], engine)
        text_df = pd.read_sql(table_queries['text_features'], engine)  # Renamed variable
        tickets = pd.read_sql(table_queries['tickets'], engine)

        # Validate data
        if core.empty or text_df.empty or tickets.empty:
            raise ValueError("❌ One or more tables are empty")

        logger.info(f"📊 Loaded: {len(core)} core, {len(text_df)} text, {len(tickets)} ticket records")

        # Create target variable based on available data
        logger.info("🎯 Creating target variable...")
        
        # Use status - assume "Closed" might indicate resolution, others might need escalation
        tickets['escalated_flag'] = (tickets['status'] != 'Closed').astype(int)
        
        logger.info("📈 Created target variable distribution:")
        logger.info(tickets['escalated_flag'].value_counts(normalize=True))

        # Optimized merging
        logger.info("🔗 Merging datasets...")
        
        # Merge in stages
        df = pd.merge(core, text_df, on="ticket_id", how="inner", validate="one_to_one")
        df = pd.merge(df, tickets[["ticket_id", "escalated_flag"]], on="ticket_id", how="inner", validate="one_to_one")
        
        logger.info(f"✅ Merged dataset shape: {df.shape}")

        # Data quality checks
        logger.info("\n🔍 Running data quality checks...")
        
        missing_stats = {col: df[col].isna().sum() for col in df.columns}
        logger.info("📊 Missing values per column:")
        for col, count in missing_stats.items():
            if count > 0:
                logger.warning(f"   {col}: {count} missing ({count/len(df)*100:.2f}%)")
        
        target_dist = df["escalated_flag"].value_counts(normalize=True)
        logger.info("📈 Escalated flag distribution:")
        for value, proportion in target_dist.items():
            logger.info(f"   {value}: {proportion:.3f}")

        logger.info(f"📋 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Optimize data types
        df = optimize_data_types(df)

        # Save to DB
        logger.info("💾 Saving features_store table...")
        df.to_sql(
            "features_store", 
            engine, 
            if_exists="replace", 
            index=False,
            chunksize=1000,
            method='multi'
        )
        
        # Create indexes - using sql_text instead of text
        with engine.begin() as conn:
            conn.execute(sql_text("CREATE INDEX IF NOT EXISTS idx_features_store_ticket_id ON features_store (ticket_id)"))
            conn.execute(sql_text("CREATE INDEX IF NOT EXISTS idx_features_store_escalated ON features_store (escalated_flag)"))
        
        logger.info("✅ Created database indexes")

        # Save to files
        os.makedirs("data", exist_ok=True)
        
        csv_path = "data/training_dataset.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"💾 Exported {csv_path}")
        
        parquet_path = "data/training_dataset.parquet"
        df.to_parquet(parquet_path, index=False, compression='snappy')
        logger.info(f"💾 Exported {parquet_path}")

        # Generate report
        generate_dataset_report(df)

        return df

    except Exception as e:
        logger.error(f"❌ Error in build_feature_store: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def optimize_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize data types to reduce memory usage"""
    df_optimized = df.copy()
    
    # Downcast numerical columns
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
    
    for col in df_optimized.select_dtypes(include=['float64']).columns:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    # Convert categorical columns
    categorical_cols = ['region']
    for col in categorical_cols:
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].astype('category')
    
    return df_optimized

def generate_dataset_report(df: pd.DataFrame):
    """Generate a comprehensive dataset report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(df),
        "features_count": len(df.columns) - 1,
        "target_distribution": df["escalated_flag"].value_counts(normalize=True).to_dict(),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    import json
    report_path = "data/dataset_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"📊 Dataset report saved to {report_path}")

if __name__ == "__main__":
    features_store = build_feature_store()
    
    if features_store is not None:
        logger.info("\n🎉 Feature store built successfully!")
        logger.info("📊 Sample rows:")
        print(features_store.head().to_string())
        
        logger.info(f"✅ Final dataset shape: {features_store.shape}")
        logger.info(f"✅ Target balance: {features_store['escalated_flag'].mean():.3f}")
    else:
        logger.error("❌ Failed to build feature store")