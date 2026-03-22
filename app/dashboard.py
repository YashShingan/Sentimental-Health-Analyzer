import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Wellness AI Pro", layout="wide")

# --- CUSTOM "BEAUTIFUL" CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #05070A; color: #F8FAFC; }
    
    /* Remove empty container gaps */
    .stVerticalBlock { gap: 1rem; }
    
    /* The Wellness Insight Card */
    .wellness-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 15px; /* Fixed the gap issue here */
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .card-icon { font-size: 26px; }
    .card-title { font-weight: 800; font-size: 13px; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 2px; }
    .card-text { font-size: 15px; opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    df = pd.read_csv("data/social_media_dataset.csv")
    model = joblib.load("models/addiction_model.pkl")
    return df, model

df, model = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.title("👤 User Profile")
    age = st.slider("Current Age", 15, 65, 24)
    usage = st.slider("Daily Screen Time (Hrs)", 0.0, 16.0, 7.5)
    sleep = st.slider("Sleep Duration (Hrs)", 2.0, 12.0, 6.5)
    platform = st.selectbox("Primary Platform", sorted(df['most_used_platform'].unique()))
    run = st.button("📊 Run Diagnostic", use_container_width=True)

if run:
    # Calculations
    input_df = pd.DataFrame({"age":[age], "avg_daily_usage_hours":[usage], "sleep_hours_per_night":[sleep]})
    p_weight = df[df['most_used_platform'] == platform]['addicted_score'].mean() / df['addicted_score'].mean()
    score = max(0, min(10, model.predict(input_df)[0] * p_weight))

    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        with st.container(border=True):
            st.caption("ADDICTION PROBABILITY INDEX")
            status_color = "#38BDF8" if score < 4 else "#FACC15" if score < 7 else "#EF4444"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=score,
                number={'suffix': "/10", 'font': {'color': 'white', 'size': 44}},
                gauge={'axis': {'range': [0, 10], 'visible': False}, 
                       'bar': {'color': status_color, 'thickness': 0.8},
                       'bgcolor': "rgba(255,255,255,0.05)"}
            ))
            fig_gauge.update_layout(height=210, margin=dict(t=0, b=0, l=20, r=20), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            sim_usage = usage * 0.5
            sim_score = max(0, min(10, model.predict(pd.DataFrame({"age":[age], "avg_daily_usage_hours":[sim_usage], "sleep_hours_per_night":[sleep]}))[0] * p_weight))
            st.info(f"⚡ **50% Cut Simulation:** Score drops to **{sim_score:.1f}/10**")

    with col2:
        st.caption("WELLNESS SUMMARY")
        avg_s = df['sleep_hours_per_night'].mean()
        avg_u = df['avg_daily_usage_hours'].mean()
        
        # BEAUTIFUL SUMMARY CARDS (NO GAPS, NO CODE LEAKS)
        st.markdown(f"""
        <div class="wellness-card">
            <div class="card-icon">🕒</div>
            <div>
                <div class="card-title" style="color: #FF4B4B;">Screen Time Analysis</div>
                <div class="card-text">You spend <b>{abs(usage-avg_u):.1f} hours {'more' if usage > avg_u else 'less'}</b> online daily than average.</div>
            </div>
        </div>
        
        <div class="wellness-card">
            <div class="card-icon">🌙</div>
            <div>
                <div class="card-title" style="color: #7D4CDB;">Rest & Recovery</div>
                <div class="card-text">You're resting <b>{abs(sleep-avg_s):.1f} hours {'more' if sleep > avg_s else 'less'}</b> than the community group.</div>
            </div>
        </div>
        
        <div class="wellness-card">
            <div class="card-icon">📱</div>
            <div>
                <div class="card-title" style="color: #00C1D4;">Platform Risk Impact</div>
                <div class="card-text"><b>{platform}</b> has a <b>{abs(((p_weight-1)*100)):.1f}%</b> statistical impact on your score.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# --- POPULATION BENCHMARKS ---
st.write("### 🧬 Population Benchmarks")
tab1, tab2 = st.tabs(["🏆 Platform Risk Ranking", "⚖️ Sleep vs Usage"])

with tab1:
    with st.container(border=True):
        p_ranking = df.groupby('most_used_platform')['addicted_score'].mean().sort_values()
        
        # BETTER NAMING: Updated Labels
        fig_bar = px.bar(p_ranking, orientation='h', color=p_ranking.values, 
                         color_continuous_scale='Blues', template="plotly_dark",
                         labels={'value': 'Addiction Risk Level (0-10)', 'most_used_platform': 'Social Media Platform'})
        
        fig_bar.update_traces(hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.2f}<extra></extra>")
        fig_bar.update_layout(height=400, coloraxis_showscale=False, 
                              xaxis_title="Addiction Risk Level (0-10)",
                              yaxis_title=None, # Cleaner Y-axis
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    with st.container(border=True):
        sample_df = df.sample(min(len(df), 700)).copy()
        sample_df['Usage (Hrs)'] = sample_df['avg_daily_usage_hours'] + np.random.uniform(-0.1, 0.1, len(sample_df))
        
        fig_scatter = px.scatter(sample_df, x="Usage (Hrs)", y="sleep_hours_per_night", 
                                 color="addicted_score", color_continuous_scale="Viridis",
                                 template="plotly_dark",
                                 labels={'Usage (Hrs)': 'Daily Media Usage (Hours)', 'sleep_hours_per_night': 'Nightly Sleep (Hours)'})
        
        fig_scatter.update_traces(marker=dict(size=6, opacity=0.8),
                                  hovertemplate="<b>Usage:</b> %{x:.1f}h<br><b>Sleep:</b> %{y:.1f}h<br><b>Score:</b> %{marker.color:.1f}<extra></extra>")
        fig_scatter.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scatter, use_container_width=True)