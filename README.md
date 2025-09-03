markdown
# Predictive Escalation Risk Model 🚨

**Proactively identify high-risk customer tickets to reduce SLA breaches and improve resolution efficiency.**

## 📌 Business Value
**Problem:** 20-30% of customer escalations violate SLAs due to late detection  
**Solution:** Machine learning model that:
- Flags high-risk tickets in real-time
- Provides explainable AI recommendations
- Simulates 20%+ reduction in SLA breaches

## 🎯 Key Metrics
| Metric                  | Target       |
|-------------------------|-------------|
| Precision (Escalation)  | ≥ 0.7       |
| Recall (Escalation)     | ≥ 0.6       |
| SLA Breach Reduction    | ≥ 20% (simulated) |
| Avg. Time-to-Escalation | ↓ 30%       |

## 🛠️ Technical Implementation

### Data Pipeline

    A[Postgres DB] --> B[Feature Engineering]
    B --> C[XGBoost Model]
    C --> D[FastAPI Endpoint]
    D --> E[Streamlit Dashboard]
# Core Features
## Feature Type	Examples
Text Analysis	OpenAI embeddings, sentiment, urgency keywords
Interaction Patterns	Transfer count, response delays
Contextual Signals	Customer tier, agent experience
# 🚀 Quick Start
## Prerequisites
Python 3.10+
PostgreSQL 14+
OpenAI API key (for embeddings)
bash
# Clone & Setup
git clone https://github.com/your-repo/predictive-escalation-risk.git
cd predictive-escalation-risk
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install
pip install -r requirements.txt
Configure Environment
bash
cp .env.example .env  # Linux/Mac
copy .env.example .env  # Windows
Edit .env:

ini
DB_HOST=localhost
DB_PORT=5432
DB_USER=your_user
DB_PASS=your_password
OPENAI_API_KEY=sk_test_123...
📂 Project Structure
text
├── data/
│   ├── sql/                 # Database schemas
│   └── samples/             # Synthetic test data
├── etl/
│   ├── feature_engineering.py
│   └── text_processing.py
├── models/
│   ├── train_model.ipynb
│   └── inference.py
├── api/                     # FastAPI endpoints
├── dashboard/               # Streamlit app
└── simulations/             # Business impact analysis
# 🌟 Key Deliverables
Jupyter Notebook: Model training with EDA

FastAPI Endpoint: /predict with risk scores

Streamlit Dashboard: Real-time risk monitoring

Simulation Report: Business impact analysis

# 📊 Sample Output
json
{
  "ticket_id": "TCKT-2024-5678",
  "risk_score": 0.83,
  "top_factors": [
    "3+ agent transfers",
    "Negative sentiment (-0.78)",
    "High-value customer"
  ],
  "llm_explanation": "This ticket shows frustration patterns similar to 82% of historical escalations..."
}
# 🤝 Stakeholder Benefits
Role	Use Case
Service Managers	Optimize agent routing
Team Leads	Prioritize high-risk tickets
Product Managers	Identify systemic issues
# 📅 Development Roadmap
Week 1: Data pipeline & EDA

Week 2: Model training & validation

Week 3: API + dashboard integration

Week 4: LLM explanations & alerting

### Key Features:
1. **Business-Focused** - Lead with ROI and stakeholder benefits
2. **Visual Hierarchy** - Icons, tables, and mermaid diagrams for scannability
3. **Technical Precision** - Clear specs without overwhelming detail
4. **Actionable Setup** - Copy-paste ready commands
5. **Storytelling** - Connects technical work to business outcomes


## 🚀 Sprint 3 — ETL Pipeline (Supabase Integration)

### Goal
Automate data cleaning and transformation from raw Supabase tables into `_clean` tables, ready for feature engineering.

### What was done
- Created `etl/load_data.py`  
- Added **table-specific cleaning functions**:
  - `clean_customers` → validates emails, normalizes regions
  - `clean_agents` → enforces numeric/non-negative `experience_years`, standardizes department names
  - `clean_tickets` → validates timestamps, normalizes status & priority
  - `clean_ticket_events` → validates event types, removes PII from descriptions
- Created `_clean` tables (`customers_clean`, `agents_clean`, `tickets_clean`, `ticket_events_clean`) in Supabase
- Inserted ETL logs into `etl_logs` table to track each run


-- Check cleaned tables
SELECT COUNT(*) FROM tickets_clean;
SELECT * FROM etl_logs ORDER BY run_time DESC;

## 🚀 Sprint 4 — Core Feature Engineering (Non-Text)

### Goal
Generate engineered features from tickets and events for model training.

### What was done
- Created `etl/feature_engineering.py`
- Extracted the following features:
  - **ticket_age_hrs** → hours since ticket was created
  - **num_transfers** → count of `"Updated"` events
  - **num_msgs_first_2h** → number of `"Commented"` events within first 2 hours
  - **avg_response_time_secs** → average gap (in seconds) between comment events
  - **region** → customer region
  - **experience_years** → agent experience
- Stored results in a new table `features_core` in Supabase


## 🚀 Sprint 5 — Text Processing Pipeline

### Goal
Generate embeddings and sentiment from ticket event text.

### What was done
- Cleaned text (removed PII, normalized case)
- Aggregated all ticket events into one document per ticket
- Generated embeddings using `sentence-transformers` (`all-MiniLM-L6-v2`)
- Computed sentiment polarity with TextBlob
- Stored results in `features_text` table

## 🚀 Sprint 6 — Feature Store Assembly

### Goal
Merge structured + text features with labels into a training dataset.

### What was done
- Loaded `features_core` (numeric features), `features_text` (embeddings + sentiment), and `tickets` (escalated_flag).
- Merged into a single table `features_store`.
- Exported dataset to `data/training_dataset.csv`.
- Performed quick data quality checks (missing values, label distribution).


## 🚀 Sprint 7 — Baseline Model Training

### 🎯 Goal
Train baseline models on the assembled feature store dataset and evaluate them.  

### ✅ What was done
- Implemented **data validation**:
  - Multiple CSV search paths
  - Fallback to Supabase `features_store` table
  - Auto-build features if missing
- Added **feature engineering**:
  - Interaction features (e.g., `ticket_age_hrs * num_transfers`)
  - Ratio features (e.g., `num_msgs_first_2h / avg_response_time_secs`)
- Implemented **time-based train/test split**
- Built **preprocessing pipelines**:
  - Numeric: median imputation + scaling  
  - Categorical: imputation + one-hot encoding  
- Trained 3 baseline models:
  - Logistic Regression
  - XGBoost
  - Random Forest
- Evaluated with:
  - Precision, Recall, F1, AUC
  - Confusion matrix
- Saved results as **artifacts**:
  - Trained models (`.joblib`)
  - Metadata (`.json`)
  - Comparison table (`model_comparison.csv`)

## 🚀 Sprint 8 — Explainability (SHAP)

### 🎯 Goal
Understand *why* the model makes predictions and provide human-readable explanations.

### ✅ What was done
- Built `models/explainability.py` with:
  - **Data + model validation** (multiple load paths, fallback to Supabase or artifacts)
  - **Preprocessing consistency** (reuse model’s preprocessor, fallback manual encoding)
  - **SHAP explainability** (TreeExplainer, LinearExplainer, KernelExplainer)
  - **Fallback feature importance** if SHAP fails
  - **Per-ticket top features** (saved in JSON)
  - **Global feature importance** (saved in CSV)
  - **Static plots**:
    - `feature_importance.png` → mean |SHAP| per feature  
    - `feature_direction.png` → whether feature increases/decreases escalation risk  
