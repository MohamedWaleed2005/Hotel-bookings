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
    background-color: #00c3ff;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
    box-shadow: 0 0 10px #00c3ff;
}
div.stButton > button:hover {
    background-color: #009ec3;
    color: white;
    box-shadow: 0 0 20px #00c3ff;
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
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Data Insights", "🔮 Prediction", "📈 Model Performance"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div style="padding:30px; border-radius:15px;
                background:linear-gradient(90deg,#0ea5e9,#1e293b); margin-bottom:20px;">
        <h2 style="color:white;">🏨 Smart Booking Risk Detection</h2>
        <p style="color:white; font-size:16px;">
            Predict cancellations and reduce revenue loss using AI & Machine Learning
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4 style="color:#00c3ff;">📋 Project Overview</h4>
            <p style="color:#d1d5db;">
            This project predicts whether a hotel booking will be <b>canceled or not</b>
            using machine learning on real-world hotel booking data.<br><br>
            The goal is to help hotels identify high-risk bookings early,
            so they can take action to reduce cancellations and revenue loss.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 Dataset Info")
        col_a, col_b = st.columns(2)
        col_a.metric("Total Rows",     f"{len(df):,}")
        col_a.metric("Total Features", f"{df.shape[1]}")
        col_b.metric("Final Model",    "XGBoost")
        col_b.metric("F1 Score",       "64.8%")
        col_b.metric("Accuracy",       "75.0%")

    st.markdown("---")
    st.subheader("🎯 Target Variable Distribution")
    target = df['is_canceled'].value_counts().reset_index()
    target.columns = ['Status', 'Count']
    target['Status'] = target['Status'].map({0: 'Not Canceled', 1: 'Canceled'})
    fig = px.pie(target, names='Status', values='Count',
                 color_discrete_sequence=['#2563EB', '#EF4444'])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Cancellation Trend by Month")
    monthly = df.groupby('arrival_date_month')['is_canceled'].mean().reset_index()
    monthly.columns = ['Month', 'Cancellation Rate']
    monthly['Cancellation Rate'] = (monthly['Cancellation Rate'] * 100).round(2)
    fig_trend = px.line(monthly, x='Month', y='Cancellation Rate',
                        markers=True, labels={'Cancellation Rate': 'Cancellation Rate (%)'},
                        color_discrete_sequence=['#00c3ff'])
    st.plotly_chart(fig_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.title("📊 Data Insights")

    # Interactive filter
    segment = st.selectbox("🔍 Filter by Market Segment", ["All"] + list(df['market_segment'].unique()))
    filtered_df = df if segment == "All" else df[df['market_segment'] == segment]
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

    # 2 — Lead Time vs Cancellation
    st.subheader("2 — Lead Time vs Cancellation")
    fig2 = px.box(filtered_df, x='is_canceled', y='lead_time', color='is_canceled',
                  labels={'is_canceled': 'Canceled', 'lead_time': 'Lead Time (days)'},
                  color_discrete_map={0: '#2563EB', 1: '#EF4444'})
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    st.info("💡 Customers with longer lead times are more likely to cancel. Hotels can reduce cancellations by sending reminders or engaging customers before arrival.")

    st.markdown("---")

    # 3 — Deposit Type vs Cancellation
    st.subheader("3 — Deposit Type vs Cancellation Rate")
    deposit = filtered_df.groupby('deposit_type')['is_canceled'].mean().reset_index()
    deposit['is_canceled'] = (deposit['is_canceled'] * 100).round(2)
    deposit.columns = ['Deposit Type', 'Cancellation Rate (%)']
    fig3 = px.bar(deposit, x='Deposit Type', y='Cancellation Rate (%)', color='Deposit Type',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#2563EB', '#F59E0B', '#EF4444'])
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(showlegend=False, yaxis_range=[0, deposit['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig3, use_container_width=True)
    st.info("💡 Deposit type has a strong relationship with cancellations. In this dataset, Non Refund bookings show the highest cancellation rate, making this feature highly informative for prediction.")

    st.markdown("---")

    # 4 — Market Segment vs Cancellation
    st.subheader("4 — Market Segment vs Cancellation Rate")
    seg = filtered_df.groupby('market_segment')['is_canceled'].mean().reset_index()
    seg['is_canceled'] = (seg['is_canceled'] * 100).round(2)
    seg.columns = ['Market Segment', 'Cancellation Rate (%)']
    seg = seg.sort_values('Cancellation Rate (%)', ascending=False)
    fig4 = px.bar(seg, x='Market Segment', y='Cancellation Rate (%)', color='Market Segment',
                  text='Cancellation Rate (%)')
    fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig4.update_layout(showlegend=False, yaxis_range=[0, seg['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig4, use_container_width=True)
    st.info("💡 Online TA and Groups have the highest cancellation rates. Hotels may apply stricter booking policies or additional confirmation for these segments.")

    st.markdown("---")

    # 5 — ADR vs Cancellation
    st.subheader("5 — ADR vs Cancellation")
    fig5 = px.box(filtered_df, x='is_canceled', y='adr', color='is_canceled',
                  labels={'is_canceled': 'Canceled', 'adr': 'Average Daily Rate (€)'},
                  color_discrete_map={0: '#2563EB', 1: '#EF4444'})
    fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)
    st.info("💡 Higher ADR bookings show a slightly higher cancellation tendency, but ADR alone is not a strong predictor.")

    st.markdown("---")

    # 6 — Special Requests vs Cancellation
    st.subheader("6 — Special Requests vs Cancellation Rate")
    special = filtered_df.groupby('total_of_special_requests')['is_canceled'].mean().reset_index()
    special['is_canceled'] = (special['is_canceled'] * 100).round(2)
    special.columns = ['Special Requests', 'Cancellation Rate (%)']
    fig6 = px.bar(special, x='Special Requests', y='Cancellation Rate (%)',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#2563EB'])
    fig6.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig6.update_layout(yaxis_range=[0, special['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig6, use_container_width=True)
    st.info("💡 Customers with more special requests are less likely to cancel, suggesting that engaged customers are more committed to their bookings.")

    st.markdown("---")

    # 7 — Repeated Guest vs Cancellation
    st.subheader("7 — Repeated Guest vs Cancellation Rate")
    repeated = filtered_df.groupby('is_repeated_guest')['is_canceled'].mean().reset_index()
    repeated['is_canceled'] = (repeated['is_canceled'] * 100).round(2)
    repeated['is_repeated_guest'] = repeated['is_repeated_guest'].map({0: 'New Guest', 1: 'Repeated Guest'})
    repeated.columns = ['Guest Type', 'Cancellation Rate (%)']
    fig7 = px.bar(repeated, x='Guest Type', y='Cancellation Rate (%)', color='Guest Type',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#2563EB', '#F59E0B'])
    fig7.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig7.update_layout(showlegend=False, yaxis_range=[0, repeated['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig7, use_container_width=True)
    st.info("💡 Returning guests rarely cancel compared to new customers. Loyalty programs can help reduce cancellations.")

    st.markdown("---")

    # 8 — Previous Cancellations vs Cancellation Rate
    st.subheader("8 — Previous Cancellations vs Cancellation Rate")
    prev = filtered_df.groupby('previous_cancellations')['is_canceled'].mean().reset_index()
    prev['is_canceled'] = (prev['is_canceled'] * 100).round(2)
    prev.columns = ['Previous Cancellations', 'Cancellation Rate (%)']
    prev = prev[prev['Previous Cancellations'] <= 10]
    fig8 = px.bar(prev, x='Previous Cancellations', y='Cancellation Rate (%)',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#EF4444'])
    fig8.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig8.update_layout(yaxis_range=[0, prev['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig8, use_container_width=True)
    st.info("💡 A history of previous cancellations is a strong indicator of future cancellation risk.")
    st.markdown("""
    > **Distribution of previous_cancellations:**
    > - 0 previous cancellations: **98.09%** of customers
    > - 1 previous cancellation: **1.59%**
    > - 2 previous cancellations: **0.13%**
    > - 3 previous cancellations: **0.07%**
    """)

    st.markdown("---")

    # 9 — Room Changed vs Cancellation
    st.subheader("9 — Room Changed vs Cancellation Rate")
    room = filtered_df.groupby('room_changed')['is_canceled'].mean().reset_index()
    room['is_canceled'] = (room['is_canceled'] * 100).round(2)
    room['room_changed'] = room['room_changed'].map({0: 'Same Room', 1: 'Room Changed'})
    room.columns = ['Room Status', 'Cancellation Rate (%)']
    fig9 = px.bar(room, x='Room Status', y='Cancellation Rate (%)', color='Room Status',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#2563EB', '#F59E0B'])
    fig9.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig9.update_layout(showlegend=False, yaxis_range=[0, room['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig9, use_container_width=True)
    st.info("💡 Room changes are associated with much lower cancellation rates. However, this feature becomes available after booking and should be interpreted carefully in real-world prediction systems.")

    st.markdown("---")

    # 10 — Top 10 Countries by Cancellation Rate
    st.subheader("10 — Top 10 Countries by Cancellation Rate")
    country_cancel = filtered_df.groupby('country').agg(
        total=('is_canceled', 'count'),
        canceled=('is_canceled', 'sum')
    ).reset_index()
    country_cancel = country_cancel[country_cancel['total'] >= 100]
    country_cancel['rate'] = (country_cancel['canceled'] / country_cancel['total'] * 100).round(2)
    country_cancel = country_cancel.nlargest(10, 'rate')
    fig10 = px.bar(country_cancel, x='rate', y='country', orientation='h',
                   text='rate', color='rate', color_continuous_scale='Reds',
                   labels={'rate': 'Cancellation Rate (%)', 'country': 'Country'})
    fig10.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig10.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig10, use_container_width=True)
    st.info("💡 Cancellation rates vary by country. Hotels can use targeted communication for customers from higher-risk regions.")

    st.markdown("---")

    # 11 — Top 10 Agents by Cancellation Rate
    st.subheader("11 — Top 10 Agents by Cancellation Rate")
    agent_cancel = filtered_df.groupby('agent').agg(
        total=('is_canceled', 'count'),
        canceled=('is_canceled', 'sum')
    ).reset_index()
    agent_cancel = agent_cancel[agent_cancel['total'] >= 50]
    agent_cancel['rate'] = (agent_cancel['canceled'] / agent_cancel['total'] * 100).round(2)
    agent_cancel = agent_cancel.nlargest(10, 'rate')
    agent_cancel['agent'] = agent_cancel['agent'].astype(float).astype(int).astype(str)
    fig11 = px.bar(agent_cancel, x='rate', y='agent', orientation='h',
                   text='rate', color='rate', color_continuous_scale='Reds',
                   labels={'rate': 'Cancellation Rate (%)', 'agent': 'Agent ID'})
    fig11.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig11.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig11, use_container_width=True)
    st.info("💡 Some agents are associated with significantly higher cancellation rates, which may reflect booking patterns or customer segments they serve.")

    st.markdown("---")

    # 12 — ADR vs Lead Time Scatter
    st.subheader("12 — ADR vs Lead Time")
    sample_df = filtered_df.sample(min(3000, len(filtered_df)), random_state=42)
    fig12 = px.scatter(sample_df, x='lead_time', y='adr',
                       color='is_canceled', opacity=0.6,
                       labels={'lead_time': 'Lead Time (days)', 'adr': 'ADR (€)', 'is_canceled': 'Canceled'},
                       color_discrete_map={0: '#2563EB', 1: '#EF4444'})
    st.plotly_chart(fig12, use_container_width=True)
    st.info("💡 High lead time combined with high ADR increases cancellation risk.")

    st.markdown("---")

    # 13 — Correlation Heatmap
    st.subheader("13 — Correlation Heatmap")
    import plotly.figure_factory as ff
    num_df = filtered_df.select_dtypes(include=np.number)
    corr = num_df.corr().round(2)
    fig13 = ff.create_annotated_heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale='Blues',
        showscale=True
    )
    fig13.update_layout(height=600)
    st.plotly_chart(fig13, use_container_width=True)
    st.info("💡 lead_time and previous_cancellations show the strongest positive correlation with is_canceled.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("🔮 Will This Booking Be Canceled?")
    st.dataframe(df.head())
    st.markdown("Fill in the booking details below and click **Predict**.")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        hotel                      = st.selectbox("Hotel Type", df['hotel'].unique())
        lead_time                  = st.number_input("Lead Time (days)", 0, 700, 30)
        arrival_date_year          = st.selectbox("Arrival Year", sorted(df['arrival_date_year'].unique()))
        arrival_date_month         = st.slider("Arrival Month", 1, 12, 6)
        arrival_date_week_number   = st.slider("Week Number", 1, 53, 25)
        arrival_date_day_of_month  = st.slider("Day of Month", 1, 31, 15)
        meal                       = st.selectbox("Meal Type", df['meal'].unique())
        total_nights               = st.number_input("Total Nights", 1, 70, 3)

    with col2:
        country                        = st.selectbox("Country", sorted(df['country'].dropna().unique()))
        market_segment                 = st.selectbox("Market Segment", df['market_segment'].unique())
        distribution_channel           = st.selectbox("Distribution Channel", df['distribution_channel'].unique())
        is_repeated_guest              = st.selectbox("Repeated Guest?", [0, 1])
        previous_cancellations         = st.number_input("Previous Cancellations", 0, 50, 0)
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", 0, 50, 0)
        booking_changes                = st.number_input("Booking Changes", 0, 20, 0)
        deposit_type                   = st.selectbox("Deposit Type", df['deposit_type'].unique())

    with col3:
        agent                       = st.number_input("Agent ID", 0, 600, 0)
        days_in_waiting_list        = st.number_input("Days in Waiting List", 0, 400, 0)
        customer_type               = st.selectbox("Customer Type", df['customer_type'].unique())
        adr                         = st.number_input("ADR (€)", 0.0, 5000.0, 100.0)
        required_car_parking_spaces = st.number_input("Parking Spaces", 0, 8, 0)
        total_of_special_requests   = st.number_input("Special Requests", 0, 5, 0)
        group_size                  = st.number_input("Group Size (adults + children + babies)", 1, 20, 2)
        weekend_ratio               = st.slider("Weekend Ratio", 0.0, 1.0, 0.3)

    room_changed   = 0
    total_bookings = previous_cancellations + previous_bookings_not_canceled

    st.markdown("---")

    if lead_time == 0:
        st.warning("⚠️ Lead time is unusually low — please verify.")
    if group_size <= 0:
        st.error("❌ Group size must be greater than 0.")

    if st.button("🔮 Predict", use_container_width=True):
        with st.spinner("Predicting..."):
            template = df.drop(columns='is_canceled').iloc[[0]].copy()
            template['hotel']                          = hotel
            template['lead_time']                      = lead_time
            template['arrival_date_year']              = arrival_date_year
            template['arrival_date_month']             = arrival_date_month
            template['arrival_date_week_number']       = arrival_date_week_number
            template['arrival_date_day_of_month']      = arrival_date_day_of_month
            template['meal']                           = meal
            template['country']                        = country
            template['market_segment']                 = market_segment
            template['distribution_channel']           = distribution_channel
            template['is_repeated_guest']              = is_repeated_guest
            template['previous_cancellations']         = previous_cancellations
            template['previous_bookings_not_canceled'] = previous_bookings_not_canceled
            template['booking_changes']                = booking_changes
            template['deposit_type']                   = deposit_type
            template['agent']                          = agent
            template['days_in_waiting_list']           = days_in_waiting_list
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

        st.markdown("---")
        st.subheader("📌 Most Similar Booking in Dataset")

        mask = (
            (df['deposit_type'] == deposit_type) &
            (df['market_segment'] == market_segment) &
            (df['hotel'] == hotel)
        )
        filtered_similar = df[mask]

        if len(filtered_similar) > 0:
            closest_idx = (filtered_similar['lead_time'] - lead_time).abs().idxmin()
            most_relevant = df.loc[closest_idx]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Your Input**")
                input_display = pd.DataFrame([{
                    'hotel': hotel,
                    'lead_time': lead_time,
                    'market_segment': market_segment,
                    'deposit_type': deposit_type,
                    'adr': adr,
                    'total_of_special_requests': total_of_special_requests,
                    'previous_cancellations': previous_cancellations,
                    'customer_type': customer_type
                }])
                st.dataframe(input_display, use_container_width=True)

            with col2:
                st.markdown("**Most Similar Booking**")
                relevant_display = most_relevant[[
                    'hotel','lead_time','market_segment','deposit_type',
                    'adr','total_of_special_requests','previous_cancellations','customer_type'
                ]].to_frame().T
                st.dataframe(relevant_display, use_container_width=True)

            if most_relevant['is_canceled'] == 1:
                st.warning("⚠️ Similar bookings in the dataset were canceled.")
            else:
                st.success("✅ Similar bookings in the dataset were NOT canceled.")
        else:
            st.info("No similar bookings found in the dataset.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📈 Model Performance")
    st.markdown("---")

    # Metrics
    st.subheader("XGBoost — Final Model Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  "75.0%")
    col2.metric("Precision", "52.4%")
    col3.metric("Recall",    "85.0%")
    col4.metric("F1 Score",  "64.8%")

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

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center; color:gray; padding:10px;">
    🏨 Hotel Booking Cancellation Prediction<br>
    Developed by <b style="color:#00c3ff;">Mohamed Waleed</b>
</div>
""", unsafe_allow_html=True)
