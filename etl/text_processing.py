import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import re
import numpy as np
from typing import List, Optional
import logging

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# DB Setup with connection pooling
# ----------------------------
load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Use connection pooling and optimize engine settings
engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800
)

# ----------------------------
# NLP Setup (lazy loading)
# ----------------------------
_embedder = None

def get_embedder():
    """Lazy load the embedding model to save memory when not used"""
    global _embedder
    if _embedder is None:
        logger.info("📥 Loading embedding model...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')  # Explicitly use CPU
    return _embedder

# ----------------------------
# Text Processing Functions
# ----------------------------
def clean_text_batch(texts: List[str]) -> List[str]:
    """Batch clean text for better performance"""
    if not texts:
        return []
    
    cleaned_texts = []
    for text in texts:
        if not isinstance(text, str):
            cleaned_texts.append("")
            continue
        # Remove emails, phone numbers, special chars
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)
        text = re.sub(r"\b\d{10,15}\b", "[PHONE]", text)
        text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)
        # Normalize case and strip
        cleaned_texts.append(text.lower().strip())
    
    return cleaned_texts

def batch_sentiment_analysis(texts: List[str]) -> List[float]:
    """Batch process sentiment analysis"""
    return [TextBlob(text).sentiment.polarity for text in texts]

# ----------------------------
# Main Feature Building Function
# ----------------------------
def build_text_features(batch_size: int = 1000) -> Optional[pd.DataFrame]:
    """
    Build text-based features from ticket events with optimized processing
    """
    try:
        logger.info("📥 Extracting ticket events...")
        
        # Only fetch necessary columns
        events = pd.read_sql(
            "SELECT ticket_id, event_description FROM ticket_events WHERE event_description IS NOT NULL", 
            engine
        )

        if events.empty:
            logger.warning("❌ No ticket events found")
            return None

        logger.info(f"📊 Processing {len(events)} events...")

        # Batch clean text
        logger.info("🧹 Cleaning text...")
        events["clean_text"] = clean_text_batch(events["event_description"].tolist())
        
        # Filter out empty texts after cleaning
        events = events[events["clean_text"].str.len() > 0]
        
        if events.empty:
            logger.warning("❌ No valid text after cleaning")
            return None

        # Aggregate per ticket
        logger.info("🔗 Aggregating text by ticket...")
        ticket_text = events.groupby("ticket_id")["clean_text"].apply(
            lambda x: " ".join(x)
        ).reset_index()

        # Batch process sentiment
        logger.info("😊 Computing sentiment in batches...")
        sentiments = batch_sentiment_analysis(ticket_text["clean_text"].tolist())
        ticket_text["sentiment"] = sentiments

        # Batch process embeddings
        logger.info("⚙️ Computing embeddings in batches...")
        embedder = get_embedder()
        
        # Process embeddings in batches to avoid memory issues
        all_embeddings = []
        texts = ticket_text["clean_text"].tolist()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embedder.encode(
                batch_texts, 
                show_progress_bar=False,  # Disable progress bar for batches
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            all_embeddings.append(batch_embeddings)
            logger.info(f"📦 Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

        # Combine all embeddings
        embeddings = np.vstack(all_embeddings)
        emb_df = pd.DataFrame(embeddings)
        emb_df.columns = [f"emb_{i}" for i in range(emb_df.shape[1])]

        # Merge features
        features_text = pd.concat([ticket_text[["ticket_id", "sentiment"]], emb_df], axis=1)

        # Save to database efficiently
        logger.info("💾 Saving features_text to DB...")
        
        # Use chunksize for large datasets
        features_text.to_sql(
            "features_text", 
            engine, 
            if_exists="replace", 
            index=False,
            chunksize=1000,
            method='multi'  # Faster insertion
        )
        
        logger.info(f"✅ Created features_text with {len(features_text)} rows and {features_text.shape[1]} columns")
        
        # Add index for faster queries
        with engine.begin() as conn:
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_features_text_ticket_id ON features_text (ticket_id)"))
        
        logger.info("📊 Sample of features:")
        logger.info(features_text[["ticket_id", "sentiment"]].head().to_string())
        
        return features_text

    except Exception as e:
        logger.error(f"❌ Error in build_text_features: {e}")
        raise

# ----------------------------
# Memory cleanup
# ----------------------------
def cleanup():
    """Clean up global resources"""
    global _embedder
    if _embedder is not None:
        del _embedder
        _embedder = None
        logger.info("🧹 Cleaned up embedding model")

if __name__ == "__main__":
    try:
        features = build_text_features(batch_size=500)  # Adjust batch size based on memory
    finally:
        cleanup()