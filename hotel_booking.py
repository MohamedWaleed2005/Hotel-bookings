import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(page_title="Hotel Booking Cancellation", page_icon="🏨", layout="wide")

@st.cache_resource
def load_artifacts():
    model    = joblib.load('xgboost.pkl')
    selector = joblib.load('feature_selector.pkl')
    return model, selector

@st.cache_data
def load_data():
    return pd.read_csv('cleaned_final.csv')

model, selector = load_artifacts()
df = load_data()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to   {opacity: 1; transform: translateY(0);}
}
.result-box { animation: fadeIn 1s ease-in-out; }

h1, h2, h3 { color: #00c3ff; }

div.stButton > button {
    background-color: #22c55e;
    color: white;
    border-radius: 0px;
    height: 3em;
    width: auto;
    padding: 0 30px;
    font-size: 18px;
    border: none;
    box-shadow: 0 0 10px #22c55e;
}
div.stButton > button:hover {
    background-color: #16a34a;
    color: white;
    box-shadow: 0 0 20px #22c55e;
}

[data-testid="stMetric"] {
    background-color: #020617;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #1e293b;
}
[data-testid="stMetricValue"] {
    font-size: 1.4rem;
    color: #00c3ff;
}

div[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
}

.info-card {
    padding: 20px;
    border-radius: 12px;
    background-color: #111827;
    border: 1px solid #1f2937;
    margin-bottom: 15px;
    transition: transform 0.3s;
}
.info-card:hover { transform: scale(1.01); }
</style>
""", unsafe_allow_html=True)

import plotly.io as pio
pio.templates.default = "plotly_dark"

st.sidebar.title("🏨 Hotel Booking")
page = st.sidebar.radio("Navigate", ["📊 Data Insights", "🔮 Prediction", "📈 Model Performance"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Data Insights":
    # ── Hero Section ──────────────────────────────────────────────
    st.markdown("""
    <div style="border-bottom: 2px solid #00c3ff; padding-bottom: 15px; margin-bottom: 20px;">
        <h1 style="color:white;">🏨 Hotel Booking Cancellation Dashboard</h1>
    </div>
    <p style="color:#d1d5db; font-size:16px;">
        A real-world hotel booking dataset collected from two hotels in Portugal (2015–2017).
        This dashboard explores booking behavior and deploys a machine learning model to predict
        <b style="color:#00c3ff;">cancellation risk</b> — a key metric for hotel revenue management.
    </p>
    """, unsafe_allow_html=True)

    # ── KPI Cards ─────────────────────────────────────────────────
    st.markdown("### Executive Summary")
    total = len(df)
    canceled = df['is_canceled'].sum()
    not_canceled = total - canceled
    cancel_rate = df['is_canceled'].mean() * 100
    avg_adr = df['adr'].mean()
    avg_lead = df['lead_time'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.markdown(f"""<div style="background:#0f2942; border:1px solid #00c3ff; border-radius:8px; padding:15px;">
        <p style="color:#00c3ff; font-size:12px; margin:0;">TOTAL BOOKINGS</p>
        <p style="color:#00c3ff; font-size:24px; font-weight:bold; margin:0;">{total:,}</p>
    </div>""", unsafe_allow_html=True)
    col2.markdown(f"""<div style="background:#0f2942; border:1px solid #00c3ff; border-radius:8px; padding:15px;">
        <p style="color:#00c3ff; font-size:12px; margin:0;">CONFIRMED</p>
        <p style="color:#00c3ff; font-size:24px; font-weight:bold; margin:0;">{not_canceled:,}</p>
    </div>""", unsafe_allow_html=True)
    col3.markdown(f"""<div style="background:#0f2942; border:1px solid #ef4444; border-radius:8px; padding:15px;">
        <p style="color:#ef4444; font-size:12px; margin:0;">CANCELED</p>
        <p style="color:#ef4444; font-size:24px; font-weight:bold; margin:0;">{canceled:,}</p>
    </div>""", unsafe_allow_html=True)
    col4.markdown(f"""<div style="background:#0f2942; border:1px solid #00c3ff; border-radius:8px; padding:15px;">
        <p style="color:#00c3ff; font-size:12px; margin:0;">AVG LEAD TIME</p>
        <p style="color:#00c3ff; font-size:24px; font-weight:bold; margin:0;">{avg_lead:.0f} days</p>
    </div>""", unsafe_allow_html=True)
    col5.markdown(f"""<div style="background:#0f2942; border:1px solid #00c3ff; border-radius:8px; padding:15px;">
        <p style="color:#00c3ff; font-size:12px; margin:0;">AVG ADR</p>
        <p style="color:#00c3ff; font-size:24px; font-weight:bold; margin:0;">€{avg_adr:.0f}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    hotel_filter = st.selectbox("🏨 Filter by Hotel Type", ["All"] + list(df['hotel'].unique()))
    filtered_df = df if hotel_filter == "All" else df[df['hotel'] == hotel_filter]
    st.caption(f"Showing {len(filtered_df):,} bookings")
    st.markdown("---")

    # 1 — Cancellation Rate
    st.subheader("1 — Overall Cancellation Rate")
    cancel_rate = filtered_df['is_canceled'].value_counts(normalize=True).reset_index()
    cancel_rate.columns = ['Status', 'Rate']
    cancel_rate['Status'] = cancel_rate['Status'].map({0: 'Not Canceled', 1: 'Canceled'})
    cancel_rate['Rate'] = (cancel_rate['Rate'] * 100).round(2)
    fig1 = px.bar(cancel_rate, x='Status', y='Rate', color='Status', text='Rate',
                  color_discrete_sequence=['#2563EB', '#EF4444'])
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig1, use_container_width=True)
    st.info("💡 About 37% of bookings are canceled, indicating a significant class imbalance.")

    st.markdown("---")

    # 2 — Lead Time + ADR (جنب بعض)
    st.subheader("2 — Lead Time & ADR vs Cancellation")
    col1, col2 = st.columns(2)
    with col1:
        fig2a = px.box(filtered_df, x='is_canceled', y='lead_time', color='is_canceled',
                       labels={'is_canceled': 'Canceled', 'lead_time': 'Lead Time (days)'},
                       color_discrete_map={0: '#2563EB', 1: '#EF4444'},
                       title="Lead Time vs Cancellation")
        fig2a.update_layout(showlegend=False)
        st.plotly_chart(fig2a, use_container_width=True)
    with col2:
        fig2b = px.box(filtered_df, x='is_canceled', y='adr', color='is_canceled',
                       labels={'is_canceled': 'Canceled', 'adr': 'ADR (€)'},
                       color_discrete_map={0: '#2563EB', 1: '#EF4444'},
                       title="ADR vs Cancellation")
        fig2b.update_layout(showlegend=False)
        st.plotly_chart(fig2b, use_container_width=True)
    st.info("💡 Customers with longer lead times are more likely to cancel. Higher ADR bookings show a slightly higher cancellation tendency.")

    st.markdown("---")

    # 3 — Deposit Type + Market Segment (جنب بعض)
    st.subheader("3 — Deposit Type & Market Segment vs Cancellation Rate")
    col1, col2 = st.columns(2)
    with col1:
        deposit = filtered_df.groupby('deposit_type')['is_canceled'].mean().reset_index()
        deposit['is_canceled'] = (deposit['is_canceled'] * 100).round(2)
        deposit.columns = ['Deposit Type', 'Cancellation Rate (%)']
        fig3a = px.bar(deposit, x='Deposit Type', y='Cancellation Rate (%)', color='Deposit Type',
                       text='Cancellation Rate (%)', title="Deposit Type")
        fig3a.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig3a.update_layout(showlegend=False, yaxis_range=[0, deposit['Cancellation Rate (%)'].max()+10])
        st.plotly_chart(fig3a, use_container_width=True)
    with col2:
        seg = filtered_df.groupby('market_segment')['is_canceled'].mean().reset_index()
        seg['is_canceled'] = (seg['is_canceled'] * 100).round(2)
        seg.columns = ['Market Segment', 'Cancellation Rate (%)']
        seg = seg.sort_values('Cancellation Rate (%)', ascending=False)
        fig3b = px.bar(seg, x='Market Segment', y='Cancellation Rate (%)', color='Market Segment',
                       text='Cancellation Rate (%)', title="Market Segment")
        fig3b.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig3b.update_layout(showlegend=False, yaxis_range=[0, seg['Cancellation Rate (%)'].max()+10])
        st.plotly_chart(fig3b, use_container_width=True)
    st.info("💡 Non Refund deposits and Online TA / Groups segments are strongly associated with cancellations.")

    st.markdown("---")

    # 4 — Special Requests + Repeated Guest + Room Changed
    st.subheader("4 — Special Requests, Repeated Guest & Room Changed")
    col1, col2, col3 = st.columns(3)
    with col1:
        special = filtered_df.groupby('total_of_special_requests')['is_canceled'].mean().reset_index()
        special['is_canceled'] = (special['is_canceled'] * 100).round(2)
        special.columns = ['Special Requests', 'Cancellation Rate (%)']
        fig4a = px.bar(special, x='Special Requests', y='Cancellation Rate (%)',
                       text='Cancellation Rate (%)', title="Special Requests",
                       color_discrete_sequence=['#2563EB'])
        fig4a.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig4a.update_layout(yaxis_range=[0, special['Cancellation Rate (%)'].max()+10])
        st.plotly_chart(fig4a, use_container_width=True)
    with col2:
        repeated = filtered_df.groupby('is_repeated_guest')['is_canceled'].mean().reset_index()
        repeated['is_canceled'] = (repeated['is_canceled'] * 100).round(2)
        repeated['is_repeated_guest'] = repeated['is_repeated_guest'].map({0: 'New', 1: 'Repeated'})
        repeated.columns = ['Guest Type', 'Cancellation Rate (%)']
        fig4b = px.bar(repeated, x='Guest Type', y='Cancellation Rate (%)', color='Guest Type',
                       text='Cancellation Rate (%)', title="Repeated Guest",
                       color_discrete_sequence=['#2563EB', '#F59E0B'])
        fig4b.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig4b.update_layout(showlegend=False, yaxis_range=[0, repeated['Cancellation Rate (%)'].max()+10])
        st.plotly_chart(fig4b, use_container_width=True)
    with col3:
        room = filtered_df.groupby('room_changed')['is_canceled'].mean().reset_index()
        room['is_canceled'] = (room['is_canceled'] * 100).round(2)
        room['room_changed'] = room['room_changed'].map({0: 'Same Room', 1: 'Changed'})
        room.columns = ['Room Status', 'Cancellation Rate (%)']
        fig4c = px.bar(room, x='Room Status', y='Cancellation Rate (%)', color='Room Status',
                       text='Cancellation Rate (%)', title="Room Changed",
                       color_discrete_sequence=['#2563EB', '#F59E0B'])
        fig4c.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig4c.update_layout(showlegend=False, yaxis_range=[0, room['Cancellation Rate (%)'].max()+10])
        st.plotly_chart(fig4c, use_container_width=True)
    st.info("💡 More special requests = lower cancellation. Repeated guests rarely cancel. Room changes are associated with lower cancellation rates.")

    st.markdown("---")

    # 5 — Previous Cancellations
    st.subheader("5 — Previous Cancellations vs Cancellation Rate")
    prev = filtered_df.groupby('previous_cancellations')['is_canceled'].mean().reset_index()
    prev['is_canceled'] = (prev['is_canceled'] * 100).round(2)
    prev.columns = ['Previous Cancellations', 'Cancellation Rate (%)']
    prev = prev[prev['Previous Cancellations'] <= 10]
    fig5 = px.bar(prev, x='Previous Cancellations', y='Cancellation Rate (%)',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#EF4444'])
    fig5.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig5.update_layout(yaxis_range=[0, prev['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig5, use_container_width=True)
    st.info("💡 A history of previous cancellations is a strong indicator of future cancellation risk.")
    st.markdown("""
    > **Distribution of previous_cancellations:**
    > - 0 previous cancellations: **98.09%** of customers
    > - 1 previous cancellation: **1.59%**
    > - 2 previous cancellations: **0.13%**
    > - 3 previous cancellations: **0.07%**
    """)

    st.markdown("---")

    # 6 — Countries + Agents (جنب بعض)
    st.subheader("6 — Top Countries & Agents by Cancellation Rate")
    col1, col2 = st.columns(2)
    with col1:
        country_cancel = filtered_df.groupby('country').agg(
            total=('is_canceled', 'count'),
            canceled=('is_canceled', 'sum')
        ).reset_index()
        country_cancel = country_cancel[country_cancel['total'] >= 100]
        country_cancel['rate'] = (country_cancel['canceled'] / country_cancel['total'] * 100).round(2)
        country_cancel = country_cancel.nlargest(10, 'rate')
        fig6a = px.bar(country_cancel, x='rate', y='country', orientation='h',
                       text='rate', color='rate', color_continuous_scale='Reds',
                       labels={'rate': 'Cancellation Rate (%)', 'country': 'Country'},
                       title="Top 10 Countries")
        fig6a.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig6a.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig6a, use_container_width=True)
    with col2:
        agent_cancel = filtered_df.groupby('agent').agg(
            total=('is_canceled', 'count'),
            canceled=('is_canceled', 'sum')
        ).reset_index()
        agent_cancel = agent_cancel[agent_cancel['total'] >= 50]
        agent_cancel['rate'] = (agent_cancel['canceled'] / agent_cancel['total'] * 100).round(2)
        agent_cancel = agent_cancel.nlargest(10, 'rate')
        agent_cancel['agent'] = agent_cancel['agent'].astype(float).astype(int).astype(str)
        fig6b = px.bar(agent_cancel, x='rate', y='agent', orientation='h',
                       text='rate', color='rate', color_continuous_scale='Reds',
                       labels={'rate': 'Cancellation Rate (%)', 'agent': 'Agent ID'},
                       title="Top 10 Agents")
        fig6b.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig6b.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig6b, use_container_width=True)
    st.info("💡 Cancellation rates vary by country and agent. Hotels can apply targeted policies for high-risk regions and agents.")

    st.markdown("---")

    # 7 — Scatter ADR vs Lead Time
    st.subheader("7 — ADR vs Lead Time")
    sample_df = filtered_df.sample(min(3000, len(filtered_df)), random_state=42)
    fig7 = px.scatter(sample_df, x='lead_time', y='adr', color='is_canceled', opacity=0.6,
                      labels={'lead_time': 'Lead Time (days)', 'adr': 'ADR (€)', 'is_canceled': 'Canceled'},
                      color_discrete_map={0: '#2563EB', 1: '#EF4444'})
    st.plotly_chart(fig7, use_container_width=True)
    st.info("💡 High lead time combined with high ADR increases cancellation risk.")

    st.markdown("---")

    # 8 — Correlation Heatmap
    st.subheader("8 — Correlation Heatmap")
    import plotly.figure_factory as ff
    num_df = filtered_df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    fig8 = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='Blues',
        showscale=True
    )
    fig8.update_layout(height=600)
    st.plotly_chart(fig8, use_container_width=True)
    st.info("💡 lead_time and previous_cancellations show the strongest positive correlation with is_canceled.")


elif page == "🔮 Prediction":
    st.title("🔮 Will This Booking Be Canceled?")
    st.dataframe(df.head())
    st.markdown("Fill in the booking details below and click **Predict**.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        hotel                  = st.selectbox("Hotel Type", df['hotel'].unique())
        lead_time              = st.number_input("Lead Time (days)", 0, 700, 30)
        arrival_date_year      = st.selectbox("Arrival Year", sorted(df['arrival_date_year'].unique()))
        arrival_date_month     = st.slider("Arrival Month", 1, 12, 6)
        meal                   = st.selectbox("Meal Type", df['meal'].unique())
        group_size             = st.number_input("Group Size", 1, 20, 2)

    with col2:
        country                        = st.selectbox("Country", sorted(df['country'].dropna().unique()))
        market_segment                 = st.selectbox("Market Segment", df['market_segment'].unique())
        distribution_channel           = st.selectbox("Distribution Channel", df['distribution_channel'].unique())
        is_repeated_guest              = st.selectbox("Repeated Guest?", [0, 1])
        previous_cancellations         = st.number_input("Previous Cancellations", 0, 50, 0)
        total_nights                   = st.number_input("Total Nights", 1, 70, 3)

    with col3:
        deposit_type                = st.selectbox("Deposit Type", df['deposit_type'].unique())
        customer_type               = st.selectbox("Customer Type", df['customer_type'].unique())
        agent                       = st.number_input("Agent ID", 0, 600, 0)
        adr                         = st.number_input("ADR (€)", 0.0, 5000.0, 100.0)
        required_car_parking_spaces = st.number_input("Parking Spaces", 0, 8, 0)
        total_of_special_requests   = st.number_input("Special Requests", 0, 5, 0)
    room_changed   = 0
    total_bookings = previous_cancellations
    previous_bookings_not_canceled = 0
    weekend_ratio  = 0.3

    st.markdown("---")

    if lead_time == 0:
        st.warning("⚠️ Lead time is unusually low — please verify.")
    if group_size <= 0:
        st.error("❌ Group size must be greater than 0.")

    _, col_center, _ = st.columns([2, 1, 2])
    with col_center:
        predict_clicked = st.button("🔮 Predict", use_container_width=True)

    if predict_clicked:
        with st.spinner("Predicting..."):
            template = df.drop(columns='is_canceled').iloc[[0]].copy()
            template['hotel']                          = hotel
            template['lead_time']                      = lead_time
            template['arrival_date_year']              = arrival_date_year
            template['arrival_date_month']             = arrival_date_month
            template['arrival_date_week_number']       = 0
            template['arrival_date_day_of_month']      = 15
            template['meal']                           = meal
            template['country']                        = country
            template['market_segment']                 = market_segment
            template['distribution_channel']           = distribution_channel
            template['is_repeated_guest']              = is_repeated_guest
            template['previous_cancellations']         = previous_cancellations
            template['previous_bookings_not_canceled'] = previous_bookings_not_canceled
            template['booking_changes']                = 0
            template['deposit_type']                   = deposit_type
            template['agent']                          = agent
            template['days_in_waiting_list']           = 0
            template['customer_type']                  = customer_type
            template['adr']                            = adr
            template['required_car_parking_spaces']    = required_car_parking_spaces
            template['total_of_special_requests']      = total_of_special_requests
            template['group_size']                     = group_size
            template['room_changed']                   = room_changed
            template['total_bookings']                 = total_bookings
            template['total_nights']                   = total_nights
            template['weekend_ratio']                  = weekend_ratio

            input_selected = selector.transform(template)
            prediction     = model.predict(input_selected)[0]
            probability    = model.predict_proba(input_selected)[0]

        st.markdown("---")
        if prediction == 1:
            prob = probability[1]*100
            st.markdown(f"""
            <div class="result-box" style="padding:30px; border-radius:15px; background-color:#2d1515;
                        border: 2px solid #ef4444; box-shadow: 0 0 20px #ef4444; text-align:center;">
                <h2 style="color:#ef4444; font-size:2rem;">❌ Booking Will Be CANCELED</h2>
                <p style="font-size:22px; color:#f87171;">
                    Cancellation Probability: <b>{prob:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)
            if prob > 80:
                st.warning("⚠️ High risk booking — consider requiring a deposit or sending an early confirmation!")
        else:
            prob = probability[0]*100
            st.markdown(f"""
            <div class="result-box" style="padding:30px; border-radius:15px; background-color:#152d1e;
                        border: 2px solid #22c55e; box-shadow: 0 0 20px #22c55e; text-align:center;">
                <h2 style="color:#22c55e; font-size:2rem;">✅ Booking Will NOT Be Canceled</h2>
                <p style="font-size:22px; color:#4ade80;">
                    Confidence: <b>{prob:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)



else:
    st.title("📈 Model Performance")
    st.markdown("---")

    # Metrics
    st.markdown("### XGBoost — Final Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("""<div style="background:#0f2942; border:1px solid #00c3ff; border-radius:8px; padding:15px;">
        <p style="color:#00c3ff; font-size:12px; margin:0;">ACCURACY</p>
        <p style="color:#00c3ff; font-size:24px; font-weight:bold; margin:0;">75.0%</p>
    </div>""", unsafe_allow_html=True)
    col2.markdown("""<div style="background:#0f2942; border:1px solid #00c3ff; border-radius:8px; padding:15px;">
        <p style="color:#00c3ff; font-size:12px; margin:0;">PRECISION</p>
        <p style="color:#00c3ff; font-size:24px; font-weight:bold; margin:0;">52.4%</p>
    </div>""", unsafe_allow_html=True)
    col3.markdown("""<div style="background:#0f2942; border:1px solid #00c3ff; border-radius:8px; padding:15px;">
        <p style="color:#00c3ff; font-size:12px; margin:0;">RECALL</p>
        <p style="color:#00c3ff; font-size:24px; font-weight:bold; margin:0;">85.0%</p>
    </div>""", unsafe_allow_html=True)
    col4.markdown("""<div style="background:#0f2942; border:1px solid #00c3ff; border-radius:8px; padding:15px;">
        <p style="color:#00c3ff; font-size:12px; margin:0;">F1 SCORE</p>
        <p style="color:#00c3ff; font-size:24px; font-weight:bold; margin:0;">64.8%</p>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")

    # Model Comparison
    st.subheader("Model Comparison")
    comparison = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost'],
        'Accuracy': [0.700, 0.660, 0.777, 0.746],
        'F1 Score': [0.581, 0.589, 0.626, 0.645]
    })
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(comparison.style.highlight_max(axis=0, color='#2563EB'), use_container_width=True)
    with col2:
        fig_comp = px.bar(comparison, x='Model', y=['Accuracy', 'F1 Score'],
                          barmode='group', color_discrete_sequence=['#2563EB', '#F59E0B'])
        fig_comp.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = np.array([[9156, 3481], [734, 3957]])
    cm_df = pd.DataFrame(cm,
                         index=['Actual: Not Canceled', 'Actual: Canceled'],
                         columns=['Predicted: Not Canceled', 'Predicted: Canceled'])
    fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues')
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")

    # Feature Importance
    st.subheader("Top Feature Importance")
    xgb_model = model.named_steps['xgboost']
    importances = xgb_model.feature_importances_
    feature_names = [
        'lead_time', 'agent', 'required_car_parking_spaces',
        'total_of_special_requests', 'room_changed',
        'market_segment_Online TA', 'deposit_type_Non Refund',
        'country (bit 5)', 'country (bit 7)'
    ]
    feat_df = pd.DataFrame({
        'Feature': feature_names[:len(importances)],
        'Importance': importances[:len(feature_names)]
    })
    feat_df = feat_df.sort_values('Importance')
    fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Blues', text='Importance')
    fig_feat.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig_feat, use_container_width=True)
