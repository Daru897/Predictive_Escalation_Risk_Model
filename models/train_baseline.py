import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, roc_auc_score, 
                            classification_report, confusion_matrix, f1_score,
                            precision_recall_curve, average_precision_score)
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Configuration
# ----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Get the project root directory dynamically
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ETL_DATA_DIR = os.path.join(PROJECT_ROOT, "etl", "data")

# ----------------------------
# Load Dataset with validation
# ----------------------------
def load_and_validate_data():
    """Load and validate the dataset from multiple possible locations"""
    possible_paths = [
        os.path.join(DATA_DIR, "training_dataset.csv"),
        os.path.join(ETL_DATA_DIR, "training_dataset.csv"),
        "data/training_dataset.csv",
        "../data/training_dataset.csv",
        "../etl/data/training_dataset.csv"
    ]
    
    for file_path in possible_paths:
        if os.path.exists(file_path):
            print(f"📁 Found dataset at: {file_path}")
            df = pd.read_csv(file_path)
            
            # Basic validation
            if df.empty:
                raise ValueError("Dataset is empty")
            
            required_columns = ["ticket_id", "escalated_flag"]
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            print(f"📊 Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
            print(f"🎯 Target distribution:\n{df['escalated_flag'].value_counts(normalize=True)}")
            
            return df
    
    # If no file found, try to load from database
    print("📁 CSV file not found, attempting to load from database...")
    return load_from_database()

def load_from_database():
    """Load data directly from database if CSV is not available"""
    try:
        from sqlalchemy import create_engine
        from dotenv import load_dotenv
        import os
        
        load_dotenv()
        
        DB_USER = os.getenv("DB_USER", "postgres")
        DB_PASS = os.getenv("DB_PASS")
        DB_HOST = os.getenv("DB_HOST")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME", "postgres")
        
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # Load from features_store table
        df = pd.read_sql("SELECT * FROM features_store", engine)
        
        if df.empty:
            # Fallback: build features from raw tables
            print("⚠️ features_store empty, building features from raw tables...")
            df = build_features_from_raw(engine)
        
        print(f"📊 Loaded from database: {df.shape[0]} samples, {df.shape[1]} features")
        return df
        
    except Exception as e:
        raise FileNotFoundError(f"Could not load data from any source: {e}")

def build_features_from_raw(engine):
    """Build features from raw tables as a fallback"""
    # This is a simplified version - you might want to import your actual feature building logic
    core = pd.read_sql("SELECT * FROM features_core", engine)
    text = pd.read_sql("SELECT ticket_id, sentiment FROM features_text", engine)
    tickets = pd.read_sql("SELECT ticket_id, status FROM tickets", engine)
    
    # Create target variable
    tickets['escalated_flag'] = (tickets['status'] != 'Closed').astype(int)
    
    # Merge
    df = pd.merge(core, text, on="ticket_id", how="inner")
    df = pd.merge(df, tickets[["ticket_id", "escalated_flag"]], on="ticket_id", how="inner")
    
    # Save for future use
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, "training_dataset.csv"), index=False)
    
    return df

# ----------------------------
# Feature Engineering
# ----------------------------
def engineer_features(df):
    """Add additional features if needed"""
    df_eng = df.copy()
    
    # Example: Create interaction features
    if all(col in df.columns for col in ['ticket_age_hrs', 'num_transfers']):
        df_eng['age_transfers_interaction'] = df_eng['ticket_age_hrs'] * df_eng['num_transfers']
    
    # Example: Create ratio features
    if all(col in df.columns for col in ['num_msgs_first_2h', 'avg_response_time_secs']):
        df_eng['msgs_per_response_ratio'] = np.where(
            df_eng['avg_response_time_secs'] > 0,
            df_eng['num_msgs_first_2h'] / df_eng['avg_response_time_secs'],
            0
        )
    
    return df_eng

# ----------------------------
# Time-based Split with validation
# ----------------------------
def time_based_split(df, test_size=TEST_SIZE):
    """Time-based split with validation"""
    # Sort by ticket_id (assuming it's time-ordered)
    df_sorted = df.sort_values("ticket_id").reset_index(drop=True)
    
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]
    
    # Ensure both classes are present in train and test
    for dataset, name in [(train_df, "Train"), (test_df, "Test")]:
        unique_classes = dataset["escalated_flag"].nunique()
        if unique_classes < 2:
            print(f"⚠️ Warning: {name} set has only {unique_classes} class(es)")
    
    X_train = train_df.drop(columns=["ticket_id", "escalated_flag"])
    y_train = train_df["escalated_flag"]
    X_test = test_df.drop(columns=["ticket_id", "escalated_flag"])
    y_test = test_df["escalated_flag"]
    
    print(f"📈 Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"📊 Train target distribution: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"📊 Test target distribution: {y_test.value_counts(normalize=True).to_dict()}")
    
    return X_train, X_test, y_train, y_test, train_df, test_df

# ----------------------------
# Preprocessing Setup
# ----------------------------
def get_feature_types(X):
    """Identify numeric and categorical features"""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"🔢 Numeric features: {numeric_features}")
    print(f"🏷️ Categorical features: {categorical_features}")
    
    return numeric_features, categorical_features

def create_preprocessor(numeric_features, categorical_features):
    """Create preprocessing pipeline for different feature types"""
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

# ----------------------------
# Model Training with pipelines
# ----------------------------
def create_model_pipelines(numeric_features, categorical_features, X_train, y_train):
    """Create optimized model pipelines with proper preprocessing"""
    
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # For small datasets, use simpler models and no early stopping
    is_small_dataset = len(X_train) < 100
    
    # Model pipelines
    pipelines = {
        'logistic_regression': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight='balanced',
                solver='liblinear'
            ))
        ]),
        
        'xgboost': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=50 if is_small_dataset else 100,  # Fewer trees for small datasets
                max_depth=3 if is_small_dataset else 4,        # Shallower trees for small datasets
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                # Remove early_stopping_rounds for small datasets
                early_stopping_rounds=None if is_small_dataset else 10
            ))
        ]),
        
        'random_forest': Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=50 if is_small_dataset else 100,  # Fewer trees for small datasets
                max_depth=3 if is_small_dataset else 5,        # Shallower trees for small datasets
                random_state=RANDOM_STATE,
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
    }
    
    return pipelines

# ----------------------------
# Model Evaluation
# ----------------------------
def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    print(f"\n{'='*50}")
    print(f"=== {model_name} ===")
    print(f"{'='*50}")
    
    # Classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Metrics
    metrics = {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }
    
    print("📈 Key Metrics:")
    for metric, value in metrics.items():
        if value is not None:
            print(f"   {metric}: {value:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🎯 Confusion Matrix:")
    print(cm)
    
    return metrics

# ----------------------------
# Save artifacts with metadata
# ----------------------------
def save_artifacts(model, model_name, metrics, feature_names):
    """Save model and metadata"""
    # Save model
    model_path = os.path.join(ARTIFACTS_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'feature_names': feature_names,
        'model_type': type(model).__name__
    }
    
    metadata_path = os.path.join(ARTIFACTS_DIR, f"{model_name}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Saved {model_name} to {model_path}")

# ----------------------------
# Main Training Function
# ----------------------------
def main():
    """Main training pipeline"""
    print("🚀 Starting model training pipeline...")
    print(f"📁 Project root: {PROJECT_ROOT}")
    print(f"📁 Data directory: {DATA_DIR}")
    
    try:
        # Load data
        df = load_and_validate_data()
        
        # Feature engineering
        df = engineer_features(df)
        
        # Time-based split
        X_train, X_test, y_train, y_test, train_df, test_df = time_based_split(df)
        
        # Get feature names and types
        feature_names = X_train.columns.tolist()
        numeric_features, categorical_features = get_feature_types(X_train)
        
        print(f"🔧 All features: {feature_names}")
        
        # Create and train models
        pipelines = create_model_pipelines(numeric_features, categorical_features, X_train, y_train)
        all_metrics = {}
        
        for model_name, pipeline in pipelines.items():
            print(f"\n🎯 Training {model_name}...")
            
            try:
                # Train model
                if model_name == 'xgboost' and len(X_train) < 100:
                    # For small datasets, fit without validation set
                    pipeline.fit(X_train, y_train)
                else:
                    pipeline.fit(X_train, y_train)
                
                # Evaluate
                metrics = evaluate_model(pipeline, X_test, y_test, model_name)
                all_metrics[model_name] = metrics
                
                # Save artifacts
                save_artifacts(pipeline, model_name, metrics, feature_names)
                
            except Exception as e:
                print(f"⚠️ Failed to train {model_name}: {e}")
                continue
        
        if not all_metrics:
            raise ValueError("❌ All models failed to train")
        
        # Compare models
        print(f"\n{'='*60}")
        print("🎯 MODEL COMPARISON")
        print(f"{'='*60}")
        
        comparison_df = pd.DataFrame(all_metrics).T
        print(comparison_df.round(4))
        
        # Save comparison
        comparison_df.to_csv(os.path.join(ARTIFACTS_DIR, "model_comparison.csv"))
        
        print(f"\n✅ All models saved in {ARTIFACTS_DIR}/")
        print("📊 Model comparison saved to artifacts/model_comparison.csv")
        
        return all_metrics
        
    except Exception as e:
        print(f"❌ Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Start training
    metrics = main()
    
    if metrics:
        best_model = max(metrics.items(), key=lambda x: x[1].get('roc_auc', 0) or x[1].get('f1_score', 0))
        print(f"\n🏆 Best model: {best_model[0]} with AUC: {best_model[1].get('roc_auc', 0):.4f}")
    else:
        print("❌ Training failed")