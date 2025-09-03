import pandas as pd
import shap
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Setup logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------
# Configuration
# ----------------------------
ARTIFACTS_DIR = "artifacts"
EXPLAIN_DIR = os.path.join(ARTIFACTS_DIR, "explainability")
os.makedirs(EXPLAIN_DIR, exist_ok=True)

# ----------------------------
# Preprocessing Functions
# ----------------------------
def preprocess_data_with_model(X: pd.DataFrame, model) -> pd.DataFrame:
    """Preprocess data using the model's preprocessing pipeline if available"""
    try:
        # If model is a pipeline with preprocessor, use it
        if hasattr(model, 'named_steps') and 'preprocessor' in model.named_steps:
            logger.info("🔧 Using model's preprocessing pipeline")
            X_processed = model.named_steps['preprocessor'].transform(X)
            
            # Get feature names from the preprocessor
            if hasattr(model.named_steps['preprocessor'], 'get_feature_names_out'):
                feature_names = model.named_steps['preprocessor'].get_feature_names_out()
                X_processed = pd.DataFrame(X_processed, columns=feature_names, index=X.index)
            else:
                X_processed = pd.DataFrame(X_processed, index=X.index)
                
        else:
            # Manual preprocessing as fallback
            logger.info("🔧 Using manual preprocessing")
            X_processed = manual_preprocessing(X)
            
        logger.info(f"✅ Preprocessed data shape: {X_processed.shape}")
        return X_processed
        
    except Exception as e:
        logger.warning(f"⚠️ Model preprocessing failed, using manual fallback: {e}")
        return manual_preprocessing(X)

def manual_preprocessing(X: pd.DataFrame) -> pd.DataFrame:
    """Manual preprocessing that matches training preprocessing"""
    X_processed = X.copy()
    
    # Handle categorical features - one-hot encode
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for feature in categorical_features:
        if feature in X_processed.columns:
            # Get all possible values to ensure consistent encoding
            all_categories = ['North', 'South', 'East', 'West']  # Known categories from your data
            for category in all_categories:
                col_name = f"{feature}_{category}"
                X_processed[col_name] = (X_processed[feature] == category).astype(int)
            X_processed = X_processed.drop(columns=[feature])
    
    # Ensure all expected features are present
    expected_features = [
        'ticket_age_hrs', 'num_transfers', 'num_msgs_first_2h', 
        'avg_response_time_secs', 'experience_years', 'sentiment',
        'region_North', 'region_South', 'region_East', 'region_West'
    ]
    
    # Add missing features with default values
    for feature in expected_features:
        if feature not in X_processed.columns:
            X_processed[feature] = 0
    
    # Select only the expected features in consistent order
    available_features = [f for f in expected_features if f in X_processed.columns]
    X_processed = X_processed[available_features]
    
    # Ensure numeric types
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    
    X_processed = X_processed.fillna(0)
    
    return X_processed

# ----------------------------
# Load Data + Model with validation
# ----------------------------
def load_data_and_model() -> Optional[tuple]:
    """Load dataset and model with comprehensive validation"""
    try:
        logger.info("📥 Loading dataset...")
        
        # Try multiple possible data paths
        data_paths = [
            "data/training_dataset.csv",
            "../data/training_dataset.csv", 
            "../etl/data/training_dataset.csv"
        ]
        
        df = None
        for path in data_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                logger.info(f"✅ Loaded data from {path}")
                break
        
        if df is None:
            raise FileNotFoundError("Could not find training dataset")
        
        if df.empty:
            raise ValueError("Dataset is empty")
        
        required_columns = ["ticket_id", "escalated_flag"]
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Load best model (try multiple candidates)
        model_candidates = [
            os.path.join(ARTIFACTS_DIR, "xgboost.joblib"),
            os.path.join(ARTIFACTS_DIR, "random_forest.joblib"), 
            os.path.join(ARTIFACTS_DIR, "logistic_regression.joblib")
        ]
        
        model = None
        model_path = None
        for candidate in model_candidates:
            if os.path.exists(candidate):
                model = joblib.load(candidate)
                model_path = candidate
                logger.info(f"✅ Loaded model from {candidate}")
                break
        
        if model is None:
            raise FileNotFoundError("No trained model found in artifacts directory")
        
        return df, model, model_path
        
    except Exception as e:
        logger.error(f"❌ Error loading data/model: {e}")
        return None

# ----------------------------
# Prepare validation data
# ----------------------------
def prepare_validation_data(df: pd.DataFrame, model, test_size: float = 0.2) -> tuple:
    """Prepare validation data with proper preprocessing"""
    # Sort by ticket_id for time-based split
    df_sorted = df.sort_values("ticket_id").reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    
    # Extract validation set
    val_df = df_sorted.iloc[split_idx:]
    X_val_raw = val_df.drop(columns=["ticket_id", "escalated_flag"])
    y_val = val_df["escalated_flag"]
    ticket_ids = val_df["ticket_id"].values
    
    # Preprocess the data using the same method as training
    X_val = preprocess_data_with_model(X_val_raw, model)
    
    logger.info(f"✅ Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    logger.info(f"📊 Validation target distribution: {y_val.value_counts(normalize=True).to_dict()}")
    logger.info(f"🔢 Features: {list(X_val.columns)}")
    
    return X_val, y_val, ticket_ids, val_df, X_val_raw

# ----------------------------
# SHAP Explainability
# ----------------------------
def compute_shap_values(model, X_val, model_type: str) -> Optional[np.ndarray]:
    """Compute SHAP values with appropriate explainer"""
    try:
        logger.info("⚙️ Computing SHAP values...")
        
        # Extract the actual classifier from pipeline if needed
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps["classifier"]
        else:
            classifier = model
        
        # For tree-based models
        if model_type.lower() in ['xgboost', 'randomforest', 'random_forest']:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_val)
            
            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # For linear models
        elif model_type.lower() in ['logistic', 'logistic_regression']:
            background = shap.sample(X_val, 3)  # Small background sample
            explainer = shap.LinearExplainer(classifier, background)
            shap_values = explainer.shap_values(X_val)
        
        else:
            # Fallback for other models
            background = shap.sample(X_val, min(3, len(X_val)))
            explainer = shap.KernelExplainer(classifier.predict_proba, background)
            shap_values = explainer.shap_values(X_val)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        logger.info(f"✅ SHAP values computed: {shap_values.shape}")
        return shap_values
        
    except Exception as e:
        logger.error(f"❌ Error computing SHAP values: {e}")
        return None

# ----------------------------
# Fallback Feature Importance
# ----------------------------
def get_fallback_importance(model, X_val, model_type: str) -> np.ndarray:
    """Get fallback feature importance when SHAP fails"""
    logger.info("🔄 Using fallback feature importance")
    
    try:
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importance = model.feature_importances_
            return np.tile(importance, (len(X_val), 1))
        
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) > 1:
                coef = model.coef_[1]  # For class 1 in binary classification
            else:
                coef = model.coef_
            importance = np.abs(coef)
            return np.tile(importance, (len(X_val), 1))
        
        else:
            # Uniform importance as last resort
            logger.warning("⚠️ Using uniform feature importance")
            return np.ones((len(X_val), X_val.shape[1])) / X_val.shape[1]
            
    except Exception as e:
        logger.error(f"❌ Fallback importance failed: {e}")
        return None

# ----------------------------
# Feature Analysis
# ----------------------------
def analyze_top_features(shap_values: np.ndarray, X_val: pd.DataFrame, ticket_ids: np.ndarray, 
                        original_X_val: pd.DataFrame, top_n: int = 3) -> List[Dict]:
    """Extract top features for each ticket"""
    logger.info(f"💾 Extracting top {top_n} features per ticket...")
    
    top_features = []
    feature_names = X_val.columns.tolist()
    
    for i, (ticket_id, row_vals) in enumerate(zip(ticket_ids, shap_values)):
        # Get top features by absolute SHAP value
        abs_vals = np.abs(row_vals)
        top_indices = np.argsort(abs_vals)[-top_n:][::-1]
        
        top_feats = []
        for idx in top_indices:
            if idx < len(feature_names):
                feature_name = feature_names[idx]
                shap_value = float(row_vals[idx])
                contribution = "increases" if shap_value > 0 else "decreases"
                
                # Map back to original feature names
                original_feature = feature_name
                if feature_name.startswith('region_'):
                    original_feature = 'region'
                
                # Get original value
                original_value = None
                if original_feature in original_X_val.columns:
                    original_value = original_X_val.iloc[i][original_feature]
                
                top_feats.append({
                    "feature": feature_name,
                    "shap_value": shap_value,
                    "contribution": contribution,
                    "feature_value": original_value,
                    "original_feature": original_feature
                })
        
        top_features.append({
            "ticket_id": int(ticket_id),
            "top_features": top_feats,
            "prediction_strength": float(np.sum(np.abs(row_vals)))
        })
    
    return top_features

def compute_global_importance(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Compute global feature importance"""
    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
        "std_shap": shap_values.std(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    
    return shap_importance

# ----------------------------
# Visualization Functions
# ----------------------------
def create_simple_plots(shap_importance: pd.DataFrame):
    """Create simple visualizations that don't require SHAP plots"""
    logger.info("📊 Generating simple plots...")
    
    # Feature importance bar plot
    plt.figure(figsize=(10, 6))
    top_features = shap_importance.head(10)
    plt.barh(range(len(top_features)), top_features['mean_abs_shap'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title("Feature Importance (Mean |SHAP|)", fontsize=14, fontweight='bold')
    plt.xlabel("Mean Absolute SHAP Value")
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, "feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature impact direction
    plt.figure(figsize=(10, 6))
    top_features = shap_importance.head(10).copy()
    top_features['color'] = top_features['mean_shap'].apply(lambda x: 'red' if x < 0 else 'blue')
    plt.barh(range(len(top_features)), top_features['mean_shap'], color=top_features['color'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.title("Feature Impact Direction (Mean SHAP)", fontsize=14, fontweight='bold')
    plt.xlabel("Mean SHAP Value (Red: decreases risk, Blue: increases risk)")
    plt.tight_layout()
    plt.savefig(os.path.join(EXPLAIN_DIR, "feature_direction.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ----------------------------
# Save Results
# ----------------------------
def save_results(top_features: List[Dict], shap_importance: pd.DataFrame):
    """Save all explainability results"""
    # Save per-ticket features
    json_path = os.path.join(EXPLAIN_DIR, "ticket_top_features.json")
    with open(json_path, "w") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(),
            "total_tickets": len(top_features),
            "ticket_features": top_features
        }, f, indent=2)
    logger.info(f"✅ Saved per-ticket features to {json_path}")
    
    # Save global importance
    csv_path = os.path.join(EXPLAIN_DIR, "global_feature_importance.csv")
    shap_importance.to_csv(csv_path, index=False)
    logger.info(f"✅ Saved global importance to {csv_path}")

# ----------------------------
# Main Function
# ----------------------------
def main():
    """Main explainability pipeline"""
    logger.info("🚀 Starting explainability pipeline...")
    
    try:
        # Load data and model
        result = load_data_and_model()
        if result is None:
            return False
        
        df, model, model_path = result
        
        # Prepare validation data with consistent preprocessing
        X_val, y_val, ticket_ids, val_df, X_val_raw = prepare_validation_data(df, model)
        
        # Compute SHAP values or use fallback
        model_type = os.path.basename(model_path).split('.')[0]
        shap_values = compute_shap_values(model, X_val, model_type)
        
        if shap_values is None:
            shap_values = get_fallback_importance(model, X_val, model_type)
        
        if shap_values is None:
            logger.error("❌ No feature importance available")
            return False
        
        # Analyze features
        top_features = analyze_top_features(shap_values, X_val, ticket_ids, X_val_raw)
        shap_importance = compute_global_importance(shap_values, X_val.columns.tolist())
        
        # Create visualizations
        create_simple_plots(shap_importance)
        
        # Save results
        save_results(top_features, shap_importance)
        
        logger.info("🎉 Explainability analysis completed successfully!")
        logger.info(f"📁 Results saved in: {EXPLAIN_DIR}/")
        
        # Show summary
        logger.info("\n📊 Top 5 Global Features:")
        for i, row in shap_importance.head().iterrows():
            direction = "↑ increases" if row['mean_shap'] > 0 else "↓ decreases"
            logger.info(f"   {row['feature']}: {row['mean_abs_shap']:.4f} ({direction})")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error in explainability pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("❌ Explainability pipeline failed")