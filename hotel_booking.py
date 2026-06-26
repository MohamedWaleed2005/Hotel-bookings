import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Hotel Booking Cancellation",
    page_icon="🏨",
    layout="wide"
)

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

st.sidebar.title("🏨 Hotel Booking")
page = st.sidebar.radio("Navigate", ["📊 EDA Dashboard", "🔮 Predict Cancellation"])

if page == "📊 EDA Dashboard":
    st.title("📊 Hotel Booking — Exploratory Data Analysis")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Bookings",    f"{len(df):,}")
    col2.metric("Cancellation Rate", f"{df['is_canceled'].mean()*100:.1f}%")
    col3.metric("Avg Lead Time",     f"{df['lead_time'].mean():.0f} days")
    col4.metric("Avg ADR",           f"€{df['adr'].mean():.0f}")

    st.markdown("---")

    st.subheader("1 — Cancellation Rate by Hotel Type")
    cancel_hotel = df.groupby('hotel')['is_canceled'].mean().reset_index()
    cancel_hotel['is_canceled'] = (cancel_hotel['is_canceled'] * 100).round(2)
    cancel_hotel.columns = ['Hotel Type', 'Cancellation Rate (%)']
    fig1 = px.bar(cancel_hotel, x='Hotel Type', y='Cancellation Rate (%)',
                  color='Hotel Type', text='Cancellation Rate (%)',
                  color_discrete_sequence=['#2563EB','#F59E0B'])
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(showlegend=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("2 — Bookings & Cancellations by Month")
    monthly = df.groupby('arrival_date_month').agg(
        Total=('is_canceled','count'),
        Canceled=('is_canceled','sum')
    ).reset_index()
    monthly['arrival_date_month'] = monthly['arrival_date_month'].astype(str)
    fig2 = px.line(monthly, x='arrival_date_month', y=['Total','Canceled'],
                   markers=True, labels={'arrival_date_month':'Month','value':'Count'},
                   color_discrete_sequence=['#2563EB','#EF4444'])
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("3 — Lead Time Distribution")
    fig3 = px.histogram(df, x='lead_time', color='is_canceled',
                        barmode='overlay', nbins=50,
                        labels={'is_canceled':'Canceled','lead_time':'Lead Time (days)'},
                        color_discrete_map={0:'#2563EB', 1:'#EF4444'})
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("4 — Cancellation Rate by Deposit Type")
    deposit = df.groupby('deposit_type')['is_canceled'].mean().reset_index()
    deposit['is_canceled'] = (deposit['is_canceled'] * 100).round(2)
    deposit.columns = ['Deposit Type', 'Cancellation Rate (%)']
    fig4 = px.bar(deposit, x='Deposit Type', y='Cancellation Rate (%)',
                  color='Deposit Type', text='Cancellation Rate (%)',
                  color_discrete_sequence=['#2563EB','#F59E0B','#EF4444'])
    fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig4.update_layout(showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("5 — ADR Distribution by Hotel Type")
    fig5 = px.box(df, x='hotel', y='adr', color='hotel',
                  labels={'hotel':'Hotel Type','adr':'Average Daily Rate (€)'},
                  color_discrete_sequence=['#2563EB','#F59E0B'])
    fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("6 — Bookings by Market Segment")
    segment = df['market_segment'].value_counts().reset_index()
    segment.columns = ['Market Segment', 'Count']
    fig6 = px.pie(segment, names='Market Segment', values='Count',
                  color_discrete_sequence=px.colors.qualitative.Bold)
    st.plotly_chart(fig6, use_container_width=True)

else:
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
        stays_in_weekend_nights    = st.number_input("Weekend Nights", 0, 20, 1)
        stays_in_week_nights       = st.number_input("Week Nights", 0, 50, 2)
        meal                       = st.selectbox("Meal Type", df['meal'].unique())

    with col2:
        country                        = st.selectbox("Country", sorted(df['country'].dropna().unique()))
        market_segment                 = st.selectbox("Market Segment", df['market_segment'].unique())
        distribution_channel           = st.selectbox("Distribution Channel", df['distribution_channel'].unique())
        is_repeated_guest              = st.selectbox("Repeated Guest?", [0, 1])
        previous_cancellations         = st.number_input("Previous Cancellations", 0, 50, 0)
        previous_bookings_not_canceled = st.number_input("Previous Bookings Not Canceled", 0, 50, 0)
        reserved_room_type             = st.selectbox("Reserved Room Type", sorted(df['reserved_room_type'].unique()))
        assigned_room_type             = st.selectbox("Assigned Room Type", sorted(df['assigned_room_type'].unique()))

    with col3:
        booking_changes             = st.number_input("Booking Changes", 0, 20, 0)
        deposit_type                = st.selectbox("Deposit Type", df['deposit_type'].unique())
        agent                       = st.number_input("Agent ID", 0, 600, 0)
        days_in_waiting_list        = st.number_input("Days in Waiting List", 0, 400, 0)
        customer_type               = st.selectbox("Customer Type", df['customer_type'].unique())
        adr                         = st.number_input("ADR (€)", 0.0, 5000.0, 100.0)
        required_car_parking_spaces = st.number_input("Parking Spaces", 0, 8, 0)
        total_of_special_requests   = st.number_input("Special Requests", 0, 5, 0)

    group_size     = st.number_input("Group Size (adults + children + babies)", 1, 20, 2)
    adults         = st.number_input("Adults", 1, 20, 2)
    children       = st.number_input("Children", 0, 10, 0)
    babies         = st.number_input("Babies", 0, 10, 0)
    room_changed   = int(reserved_room_type != assigned_room_type)
    total_bookings = previous_cancellations + previous_bookings_not_canceled

    st.markdown("---")

    if st.button("🔮 Predict", use_container_width=True):
        input_data = pd.DataFrame([{
            'hotel': hotel,
            'lead_time': lead_time,
            'arrival_date_year': arrival_date_year,
            'arrival_date_month': arrival_date_month,
            'arrival_date_week_number': arrival_date_week_number,
            'arrival_date_day_of_month': arrival_date_day_of_month,
            'stays_in_weekend_nights': stays_in_weekend_nights,
            'stays_in_week_nights': stays_in_week_nights,
            'adults': adults,
            'children': children,
            'babies': babies,
            'meal': meal,
            'country': country,
            'market_segment': market_segment,
            'distribution_channel': distribution_channel,
            'is_repeated_guest': is_repeated_guest,
            'previous_cancellations': previous_cancellations,
            'previous_bookings_not_canceled': previous_bookings_not_canceled,
            'reserved_room_type': reserved_room_type,
            'assigned_room_type': assigned_room_type,
            'booking_changes': booking_changes,
            'deposit_type': deposit_type,
            'agent': agent,
            'days_in_waiting_list': days_in_waiting_list,
            'customer_type': customer_type,
            'adr': adr,
            'required_car_parking_spaces': required_car_parking_spaces,
            'total_of_special_requests': total_of_special_requests,
            'group_size': group_size,
            'room_changed': room_changed,
            'total_bookings': total_bookings
        }])

        input_selected = selector.transform(input_data)
        prediction     = model.predict(input_selected)[0]
        probability    = model.predict_proba(input_selected)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(f"❌ This booking is likely to be **CANCELED** (Confidence: {probability[1]*100:.1f}%)")
        else:
            st.success(f"✅ This booking is likely to **NOT be canceled** (Confidence: {probability[0]*100:.1f}%)")
