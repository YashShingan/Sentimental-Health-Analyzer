import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import streamlit.components.v1 as components
import textwrap

# --- PAGE SETUP ---
st.set_page_config(page_title="Wellness AI Pro", layout="wide")

# --- CUSTOM CSS (High Impact Typography & Bento Grid) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; background-color: #05070A; color: #F8FAFC; }
    
    /* Bento Box Grid for Insights */
    .insight-container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 25px; }
    .insight-card { 
        background: rgba(255, 255, 255, 0.03); 
        padding: 25px; 
        border-radius: 18px; 
        border: 1px solid rgba(255,255,255,0.08);
    }
    .insight-header { font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 12px; opacity: 0.6; }
    .insight-value { font-size: 22px; font-weight: 700; line-height: 1.3; }
    .full-width { grid-column: span 2; }
    
    /* Reverted Simulation Style */
    .sim-box {
        background: rgba(0, 193, 212, 0.05);
        border: 1px dashed #00C1D4;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOADING ASSETS ---
@st.cache_resource
def load_assets():
    df = pd.read_csv("data/social_media_dataset.csv")
    model = joblib.load("models/addiction_model.pkl")
    model_cols = joblib.load("models/model_columns.pkl")
    return df, model, model_cols

df, model, model_cols = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.title("👤 User Profile")
    age = st.slider("Current Age", 15, 65, 25)
    usage = st.slider("Daily Screen Time (Hrs)", 0.0, 16.0, 6.0)
    sleep = st.slider("Sleep Duration (Hrs)", 2.0, 12.0, 7.0)
    platform = st.selectbox("Primary Platform", sorted(df['most_used_platform'].unique()))
    run = st.button("📊 Run Diagnostic", use_container_width=True)

if run:
   # --- MATH LOGIC ---
    input_data = pd.DataFrame(0, index=[0], columns=model_cols)
    input_data["age"], input_data["avg_daily_usage_hours"], input_data["sleep_hours_per_night"] = age, usage, sleep
    if "usage_sleep_ratio" in input_data.columns:
        input_data["usage_sleep_ratio"] = usage / (sleep + 1)
    
    platform_col = f"most_used_platform_{platform}"
    if platform_col in input_data.columns:
        input_data[platform_col] = 1
        
    input_data = input_data[model_cols]
    
    # 1. Get the base prediction from the model
    base_score = model.predict(input_data)[0]

    # 2. Calculate the Platform Risk Factor directly from the CSV
    global_avg = df['addicted_score'].mean()
    platform_avg = df[df['most_used_platform'] == platform]['addicted_score'].mean()
    
    # Variable fixed here to match your later usage
    platform_impact_pct = (platform_avg - global_avg) / global_avg
    
    # 3. Apply the platform's statistical impact to the base score
    adjusted_score = base_score * (1 + platform_impact_pct)
    score = max(0.0, min(10.0, adjusted_score))

    # --- 50% CUT SIMULATION ---
    sim_data = input_data.copy()
    sim_data["avg_daily_usage_hours"] = usage * 0.5
    if "usage_sleep_ratio" in sim_data.columns:
        sim_data["usage_sleep_ratio"] = (usage * 0.5) / (sleep + 1)
        
    base_sim_score = model.predict(sim_data)[0]
    sim_score = max(0.0, min(10.0, base_sim_score * (1 + platform_impact_pct)))
    score_drop = score - sim_score

    # UI Logic
    r_color = "#10B981" if score < 4 else "#F59E0B" if score < 7 else "#EF4444"
    status_label = "Healthy" if score < 4 else "Moderate Risk" if score < 7 else "High Risk"

    # --- TOP SECTION ---
    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.caption("YOUR WELLNESS SCORE")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=score,
            number={'suffix': "/10", 'font': {'color': r_color, 'size': 50}},
            gauge={
                'axis': {'range': [0, 10], 'tickwidth': 1, 'tickcolor': "white", 'dtick': 2},
                'bar': {'color': r_color, 'thickness': 0.25},
                'bgcolor': "rgba(255,255,255,0.05)",
                'steps': [
                    {'range': [0, 4], 'color': "rgba(16, 185, 129, 0.2)"}, 
                    {'range': [4, 7], 'color': "rgba(245, 158, 11, 0.2)"}, 
                    {'range': [7, 10], 'color': "rgba(239, 68, 68, 0.2)"}
                ],
                'threshold': {'line': {'color': r_color, 'width': 5}, 'thickness': 0.8, 'value': score}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=0, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"""
        <div class="sim-box">
            <span style="color:#00C1D4; font-weight:800;">💡 WHAT IF YOU CUT USE IN HALF?</span><br>
            Your score would drop to <b>{sim_score:.1f}</b> — that's a <span style="color:#10B981;">{score_drop:.1f} point improvement!</span>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.caption("PERSONALIZED INSIGHTS")
        peer_avg = df[(df['age'] >= age-2) & (df['age'] <= age+2)]['addicted_score'].mean()
        
        st.markdown(f"""
        <div class="insight-container">
            <div class="insight-card">
                <div class="insight-header" style="color:{r_color}">Current Standing</div>
                <div class="insight-value">Your current habits are in the <span style="color:{r_color}">{status_label}</span> range.</div>
            </div>
            <div class="insight-card">
                <div class="insight-header">How you compare</div>
                <div class="insight-value">Most people your age score around <b>{peer_avg:.1f}</b>. You're doing <b>{abs(score-peer_avg):.1f}</b> points {'better' if score < peer_avg else 'worse'} than average.</div>
            </div>
            <div class="insight-card full-width" style="border-left: 6px solid {r_color};">
                <div class="insight-header" style="opacity: 1;">A simple tip for you</div>
                <div class="insight-value" style="font-size: 18px; font-weight: 400; opacity: 0.9;">
                    {'You have a great balance! Just keep doing what you are doing.' if score < 4 else 
                     'Putting your phone away 20 minutes earlier tonight could help you reach the "Healthy" zone.' if score < 7 else 
                     'To help your brain rest, try turning on "Grayscale" mode and avoid screens for an hour before bed.'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- RECOMMENDATION SECTION ---
    peer_data = df[(df['age'] >= age-3) & (df['age'] <= age+3)]['addicted_score']
    peer_avg_calc = peer_data.mean() if not peer_data.empty else 5.0
    
    sleep_deficit = max(0, 8.0 - sleep)
    usage_ratio = (usage / sleep) if sleep > 0 else usage
    recovery_debt = "High" if sleep_deficit > 1.5 or usage_ratio > 1.2 else "Low"
    focus_score = max(10, 100 - int((usage * 8) - (sleep * 2)))
    impact_display = abs(platform_impact_pct * 100)

    if score < 4:
        rec_status, rec_color, advice = "Optimal", "#10B981", "Your neural recovery is excellent. Maintain this 1:1 balance."
    elif score < 7:
        rec_status, rec_color, advice = "Strained", "#F59E0B", "Usage is outpacing rest. A 20% reduction in screen time would reset your focus."
    else:
        rec_status, rec_color, advice = "Critical", "#EF4444", "High dopamine dependency detected. Immediate 'Digital Detox' required."

    st.markdown("---")
    
    # Indentation fix using textwrap.dedent
    rec_html = f"""
    <div style="background: linear-gradient(145deg, {rec_color}15, #0a0c10); border: 1px solid {rec_color}33; border-radius: 20px; padding: 30px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 25px;">
            <div>
                <h3 style="margin: 0; color: {rec_color}; font-size: 24px; font-weight: 800;">🎯 Deep-Dive Analysis</h3>
                <p style="margin: 5px 0 0 0; opacity: 0.6; font-size: 14px;">Personalized Bio-Digital Feedback Loop</p>
            </div>
            <div style="background: {rec_color}; color: #000; padding: 6px 16px; border-radius: 30px; font-weight: 900; font-size: 12px; letter-spacing: 1px;">
                STATUS: {rec_status.upper()}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-bottom: 25px;">
            <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05);">
                <p style="font-size: 10px; opacity: 0.5; text-transform: uppercase; margin: 0; letter-spacing: 1px;">Recovery Debt</p>
                <p style="font-size: 22px; font-weight: 700; margin: 5px 0; color: {rec_color if recovery_debt == 'Low' else '#EF4444'};">{recovery_debt}</p>
                <p style="font-size: 12px; opacity: 0.4; margin: 0;">{sleep_deficit:.1f}h sleep deficit</p>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05);">
                <p style="font-size: 10px; opacity: 0.5; text-transform: uppercase; margin: 0; letter-spacing: 1px;">Neural Focus</p>
                <p style="font-size: 22px; font-weight: 700; margin: 5px 0;">{focus_score}%</p>
                <p style="font-size: 12px; opacity: 0.4; margin: 0;">Estimated attention span</p>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05);">
                <p style="font-size: 10px; opacity: 0.5; text-transform: uppercase; margin: 0; letter-spacing: 1px;">Platform Weight</p>
                <p style="font-size: 22px; font-weight: 700; margin: 5px 0;">{impact_display:.1f}%</p>
                <p style="font-size: 12px; opacity: 0.4; margin: 0;">Impact of {platform}</p>
            </div>
            <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05);">
                <p style="font-size: 10px; opacity: 0.5; text-transform: uppercase; margin: 0; letter-spacing: 1px;">Peer Delta</p>
                <p style="font-size: 22px; font-weight: 700; margin: 5px 0;">{score - peer_avg_calc:+.1f}</p>
                <p style="font-size: 12px; opacity: 0.4; margin: 0;">vs. {age}yo average</p>
            </div>
        </div>
        <div style="background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border-left: 4px solid {rec_color};">
            <h4 style="margin: 0 0 8px 0; font-size: 14px; color: {rec_color}; text-transform: uppercase;">🚀 Strategic Action Plan</h4>
            <p style="margin: 0; font-size: 16px; line-height: 1.5; opacity: 0.9;">{advice}</p>
        </div>
    </div>
    """
    st.markdown(textwrap.dedent(rec_html), unsafe_allow_html=True)

# --- POPULATION INSIGHTS ---
st.write("---")
st.write("### 🧬 Population Insights")
tab1, tab2 = st.tabs(["🏆 Global Rankings", "⚖️ Sleep vs Usage"])

with tab1:
    with st.container(border=True):
        p_ranking = df.groupby('most_used_platform')['addicted_score'].mean().sort_values()
        
        fig_bar = px.bar(p_ranking, orientation='h', color=p_ranking.values, 
                         color_continuous_scale='Blues', template="plotly_dark",
                         labels={'value': 'Addiction Risk Level (0-10)', 'most_used_platform': 'Social Media Platform'})
        
        fig_bar.update_traces(hovertemplate="<b>%{y}</b><br>Risk Score: %{x:.2f}<extra></extra>")
        fig_bar.update_layout(height=400, coloraxis_showscale=False, 
                              xaxis_title="Addiction Risk Level (0-10)",
                              yaxis_title=None, 
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