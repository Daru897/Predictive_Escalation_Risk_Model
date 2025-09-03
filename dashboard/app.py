import streamlit as st
import pandas as pd
import requests
import numpy as np
from datetime import datetime
import time
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
import os

# Add the parent directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ----------------------------
# Configuration
# ----------------------------
st.set_page_config(
    page_title="Escalation Risk Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# API Configuration - with fallback
try:
    from config import API_URL
except ImportError:
    API_URL = "http://127.0.0.1:8000"

CACHE_TTL = 300  # 5 minutes cache

# ----------------------------
# Caching and Session State
# ----------------------------
@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load and cache prediction data"""
    try:
        # Try to load from API first
        response = requests.get(f"{API_URL}/predictions", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data.get('predictions', []))
    except requests.RequestException as e:
        st.sidebar.warning(f"⚠️ API unavailable: {e}. Using demo data.")
    
    # Fallback to demo data with more realistic information
    demo_data = {
        "ticket_id": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "risk_score": [0.75, 0.42, 0.88, 0.35, 0.92, 0.28, 0.67, 0.51, 0.83, 0.39],
        "llm_explanation": [
            "The ticket is old and has many transfers, indicating potential escalation risk.",
            "The sentiment is neutral and response time is within acceptable limits.",
            "High negative sentiment and long delays significantly increase escalation risk.",
            "Low activity and normal response patterns suggest minimal risk.",
            "Multiple customer complaints and slow resolution times indicate high risk.",
            "Standard support ticket with no concerning patterns detected.",
            "Moderate risk due to extended ticket age and customer frustration signs.",
            "Balanced risk factors with some concerning patterns emerging.",
            "Urgent customer demands and complex technical issues increase risk substantially.",
            "Normal support patterns with satisfactory customer engagement."
        ],
        "status": ["Open", "Closed", "Open", "In Progress", "Open", "Closed", "In Progress", "Open", "Open", "Closed"],
        "priority": ["High", "Low", "High", "Medium", "High", "Low", "Medium", "Medium", "High", "Low"],
        "created_at": pd.date_range('2024-01-01', periods=10, freq='D'),
        "agent_id": [1, 2, 1, 3, 2, 3, 1, 2, 3, 1],
        "customer_tier": ["Enterprise", "Basic", "Enterprise", "Premium", "Basic", "Premium", "Enterprise", "Basic", "Premium", "Basic"],
        "response_time_hrs": [48.5, 12.3, 72.1, 8.5, 96.2, 6.7, 36.8, 24.1, 84.3, 18.9]
    }
    return pd.DataFrame(demo_data)

@st.cache_data(ttl=CACHE_TTL)
def get_ticket_details(ticket_id: int) -> dict:
    """Get detailed ticket information from API"""
    try:
        response = requests.get(f"{API_URL}/explanations/{ticket_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    
    # Fallback mock data
    return {
        "explanations": [
            {"feature": "ticket_age_hrs", "shap_value": 0.15, "feature_value": 48.5},
            {"feature": "num_transfers", "shap_value": 0.12, "feature_value": 3},
            {"feature": "sentiment", "shap_value": -0.08, "feature_value": -0.2}
        ]
    }

# ----------------------------
# Helper Functions
# ----------------------------
def format_risk_score(score: float) -> str:
    """Format risk score with color coding"""
    if score >= 0.7:
        return f"🔴 {score:.2f}"
    elif score >= 0.4:
        return f"🟡 {score:.2f}"
    else:
        return f"🟢 {score:.2f}"

def create_risk_gauge(score: float) -> go.Figure:
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 20}},
        delta={'reference': 0.5, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.4], 'color': 'lightgreen'},
                {'range': [0.4, 0.7], 'color': 'lightyellow'},
                {'range': [0.7, 1], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0.7
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def check_api_health() -> bool:
    """Check if API is available"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        return response.status_code == 200
    except requests.RequestException:
        return False

# ----------------------------
# Main Dashboard
# ----------------------------
def main():
    st.title("📊 Escalation Risk Dashboard")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar Filters
    st.sidebar.header("🔧 Filters & Controls")
    
    risk_threshold = st.sidebar.slider(
        "Risk Score Threshold", 
        0.0, 1.0, 0.5, 0.05,
        help="Set the minimum risk score to display"
    )
    
    status_filter = st.sidebar.multiselect(
        "Status",
        options=df['status'].unique(),
        default=df['status'].unique(),
        help="Filter by ticket status"
    )
    
    priority_filter = st.sidebar.multiselect(
        "Priority",
        options=df['priority'].unique(),
        default=df['priority'].unique(),
        help="Filter by ticket priority"
    )
    
    # Apply filters
    filtered_df = df[
        (df['risk_score'] >= risk_threshold) &
        (df['status'].isin(status_filter)) &
        (df['priority'].isin(priority_filter))
    ].copy()
    
    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Tickets", len(df))
    
    with col2:
        risk_percentage = len(filtered_df) / len(df) * 100 if len(df) > 0 else 0
        st.metric("At Risk", len(filtered_df), f"{risk_percentage:.1f}%")
    
    with col3:
        avg_risk = filtered_df['risk_score'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg Risk Score", f"{avg_risk:.2f}")
    
    with col4:
        high_risk = len(filtered_df[filtered_df['risk_score'] >= 0.7])
        st.metric("High Risk", high_risk)
    
    st.markdown("---")
    
    # Main Content
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📋 Tickets at Risk")
        
        if not filtered_df.empty:
            # Enhanced dataframe display
            display_df = filtered_df[['ticket_id', 'risk_score', 'status', 'priority', 'created_at']].copy()
            display_df['risk_score'] = display_df['risk_score'].apply(format_risk_score)
            display_df['created_at'] = display_df['created_at'].dt.strftime('%Y-%m-%d')
            
            st.dataframe(
                display_df.sort_values('risk_score', ascending=False),
                use_container_width=True,
                height=400,
                hide_index=True
            )
        else:
            st.info("No tickets match the current filters. Adjust the threshold or filters to see results.")
    
    with col_right:
        st.subheader("📈 Risk Distribution")
        
        # Risk distribution chart
        risk_bins = pd.cut(df['risk_score'], bins=[0, 0.4, 0.7, 1], labels=['Low', 'Medium', 'High'])
        risk_counts = risk_bins.value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Ticket Details Section
    st.subheader("🔍 Ticket Details")
    
    if not filtered_df.empty:
        ticket_id = st.selectbox(
            "Select Ticket for Details",
            options=filtered_df['ticket_id'].tolist(),
            format_func=lambda x: f"Ticket {x}"
        )
        
        if ticket_id:
            ticket_data = df[df['ticket_id'] == ticket_id].iloc[0]
            details = get_ticket_details(ticket_id)
            
            col_detail_left, col_detail_right = st.columns([1, 2])
            
            with col_detail_left:
                st.plotly_chart(create_risk_gauge(ticket_data['risk_score']), use_container_width=True)
                
                # Basic info
                st.info(f"""
                **Ticket Information:**
                - **Status:** {ticket_data['status']}
                - **Priority:** {ticket_data['priority']}
                - **Created:** {ticket_data['created_at'].strftime('%Y-%m-%d')}
                - **Agent ID:** {ticket_data.get('agent_id', 'N/A')}
                - **Customer Tier:** {ticket_data.get('customer_tier', 'N/A')}
                """)
            
            with col_detail_right:
                # LLM Explanation
                st.write("**🤖 AI Explanation:**")
                st.info(ticket_data['llm_explanation'])
                
                # Feature importance if available
                if details and 'explanations' in details:
                    st.write("**📊 Top Contributing Factors:**")
                    features_df = pd.DataFrame(details['explanations'])
                    if not features_df.empty:
                        # Create horizontal bar chart for feature importance
                        fig_features = px.bar(
                            features_df,
                            x='shap_value',
                            y='feature',
                            orientation='h',
                            color='shap_value',
                            color_continuous_scale='RdYlGn_r',
                            title="Feature Impact on Risk Score"
                        )
                        fig_features.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig_features, use_container_width=True)
    
    # Export and Actions
    st.sidebar.markdown("---")
    st.sidebar.subheader("💾 Export & Actions")
    
    # Export buttons
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Export All Data (CSV)",
        data=csv_data,
        file_name=f"escalation_risk_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        help="Download complete dataset as CSV"
    )
    
    if not filtered_df.empty:
        filtered_csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Export Filtered Data (CSV)",
            data=filtered_csv,
            file_name=f"filtered_risk_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            help="Download filtered dataset as CSV"
        )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # API Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🌐 API Status")
    
    if check_api_health():
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.warning("⚠️ API Offline - Using Demo Data")

# ----------------------------
# Run the app properly
# ----------------------------
if __name__ == "__main__":
    # Check if running with streamlit run
    if 'streamlit' in sys.modules:
        main()
    else:
        print("Please run this script with: streamlit run app.py")
        print("Alternatively, you can run it directly for development:")
        
        # Simulate streamlit environment for development
        try:
            import streamlit.web.bootstrap
            import streamlit.runtime.scriptrunner.magic_funcs
            main()
        except Exception as e:
            print(f"Error running dashboard: {e}")
            print("Please install required packages: pip install streamlit pandas requests plotly")