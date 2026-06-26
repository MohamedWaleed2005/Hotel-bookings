# Hotel Booking Cancellation Prediction

## Objective
The goal of this project is to predict whether a hotel booking will be canceled based on customer, booking, and reservation information.

## Dataset
Hotel Booking Demand Dataset
| Column Name | Description |
|------------|-------------|
| hotel | Type of hotel (Resort Hotel or City Hotel). |
| is_canceled | Indicates whether the booking was canceled (1) or not (0). |
| lead_time | Number of days between booking date and arrival date. |
| arrival_date_year | Year of arrival. |
| arrival_date_month | Month of arrival. |
| arrival_date_week_number | Week number of the year for arrival. |
| arrival_date_day_of_month | Day of the month of arrival. |
| stays_in_weekend_nights | Number of weekend nights booked. |
| stays_in_week_nights | Number of weekday nights booked. |
| adults | Number of adults included in the booking. |
| children | Number of children included in the booking. |
| babies | Number of babies included in the booking. |
| meal | Type of meal booked. |
| country | Country of origin of the guest. |
| market_segment | Market segment through which the booking was made. |
| distribution_channel | Booking distribution channel. |
| is_repeated_guest | Indicates whether the guest has stayed before. |
| previous_cancellations | Number of previous canceled bookings by the customer. |
| previous_bookings_not_canceled | Number of previous bookings not canceled. |
| reserved_room_type | Room type originally reserved by the guest. |
| assigned_room_type | Room type assigned by the hotel. |
| booking_changes | Number of changes made to the booking. |
| deposit_type | Type of deposit made for the booking. |
| agent | ID of the travel agent that made the booking. |
| company | ID of the company responsible for the booking. |
| days_in_waiting_list | Number of days the booking was on the waiting list. |
| customer_type | Type of customer (Transient, Contract, Group, etc.). |
| adr | Average Daily Rate (average revenue per occupied room per day). |
| required_car_parking_spaces | Number of parking spaces requested by the guest. |
| total_of_special_requests | Number of special requests made by the guest. |
| reservation_status | Final reservation status (Canceled, Check-Out, No-Show). |
| reservation_status_date | Date when the reservation status was last updated. |
# Import Libraries
import pandas as pd
import numpy as np 
import plotly.express as px
import datetime 
# Load Dataset
df=pd.read_csv('hotel_bookings.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df.head()
# Data Understanding
df.info()
(df.isna().mean()*100).sort_values(ascending=False) # Check missing values
df.describe()
# Data Cleaning
drop=['company','reservation_status_date','reservation_status'] # unnecessary columns

df.drop(columns=drop,inplace=True)
df.duplicated().sum() # check duplicates
df.drop_duplicates(inplace=True) #remove duplicates
df.duplicated().sum()
df.info()
#check outliers

from datasist.structdata import detect_outliers
num=df.select_dtypes(include='number').drop(columns='is_canceled').columns
for col in num:
    outliers=detect_outliers(data=df,n=0,features=[col])
    print(col)
    
    print(len(outliers)/df.shape[0]*100)
    print('*'*100)
df[df['adr']<0].shape # adr column  include negative numbers
df=df[~(df['adr']<0)]
df.describe(include='O')
# check categorical columns
cat=df.select_dtypes(include='O').columns
for col in cat:
    print(col)
    print(df[col].nunique())
    print(df[col].unique())
    print('*'*100)
for col in cat:
    print(df[col].value_counts(normalize=True)*100)
    print('*'*100)
df=df[~(df[cat]=='Undefined').any(axis=1)]  # Undefined handling
df['market_segment'].value_counts(normalize=True)*100
df.reset_index(inplace=True,drop=True)
# Feature Engineering
df['group_size'] = df['adults'] + df['children'] + df['babies']  # Create total guests feature

df['room_changed']=(df['reserved_room_type'] != df['assigned_room_type']).astype(int)  # Check if room assignment changed


df['total_bookings']=df['previous_cancellations']+df['previous_bookings_not_canceled']   # Total previous bookings

df['arrival_date_month']=pd.to_datetime(df['arrival_date_month'],format='%B').dt.month

num=df.select_dtypes(include='number').drop(columns='is_canceled').columns

# Total nights (أهم feature)
df["total_nights"] = df["stays_in_week_nights"] + df["stays_in_weekend_nights"]

# نسبة الweekend (feature ذكية)
df["weekend_ratio"] = df["stays_in_weekend_nights"] / (df["total_nights"] + 1)
df.drop(["stays_in_week_nights", "stays_in_weekend_nights"], axis=1, inplace=True)

df.drop(["children", "babies", "adults"], axis=1, inplace=True)
df.drop(["assigned_room_type", "reserved_room_type"], axis=1, inplace=True)

import pycountry

def code_to_name(code):
    
    try:
        x=pycountry.countries.get(alpha_3=code)          # get countries name
        return x.name
    except:
        return np.nan
        

df['country']=df['country'].apply(code_to_name)
df.head()
df.to_csv('cleaned_final.csv',index=False)
# Data Preprocessing
x=df.drop(columns='is_canceled')
y=df['is_canceled']
be_nom=['country','reserved_room_type','assigned_room_type']
ohe_nom=['hotel','meal','market_segment','distribution_channel','deposit_type','customer_type']

## Scaling
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
rs=RobustScaler()
k=KNNImputer(n_neighbors=5)
num_pip=Pipeline(steps=[('Impute',k),('scaling',rs)])
num_pip
## Encoding
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(drop='first',sparse_output=False)
ohe_pip=Pipeline(steps=[("one hot",ohe)])
ohe_pip
from category_encoders import BinaryEncoder
from sklearn.impute import SimpleImputer
si=SimpleImputer(strategy='most_frequent')
be=BinaryEncoder()
be_pip=Pipeline(steps=[('simple imputer',si),('binary encoding',be)])
be_pip
## Pipeline
from sklearn.compose import ColumnTransformer
pre_pipe=ColumnTransformer(transformers=[('num cols',num_pip,num),
                                         ('one hot encoder',ohe_pip,ohe_nom),
                                         ('binary encoder',be_pip,be_nom)]
                                         ,remainder='passthrough')
pre_pipe

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as pip

   
## Feature Selection

from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline


dt_pipe = Pipeline(steps=[
    ('preprocessing', pre_pipe),
    ('model', DecisionTreeClassifier(max_depth=5))
])

dt_pipe.fit(x, y)


feature_names = dt_pipe[:-1].get_feature_names_out()
importances = dt_pipe['model'].feature_importances_

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print(importance_df.head(20))
selected_features = importance_df[importance_df['importance'] > 0]['feature'].tolist()
print(f"Selected {len(selected_features)} features")


from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline as pip

selector = pip(steps=[
    ('preprocessing', pre_pipe),
    ('feature_selection', SelectFromModel(DecisionTreeClassifier(max_depth=5), threshold=0.01))])

x_selected = selector.fit_transform(x, y)
print('Features after selection',x_selected.shape[1])
# Model Selection
model_zoo = [
    ('logistic', LogisticRegression()),
    ('decision tree', DecisionTreeClassifier(max_depth=5)),
    ('random forest', RandomForestClassifier()),
    ('xgboost', XGBClassifier())
]

for model in model_zoo:
    model_pipe = pip(steps=[
        ('smote', SMOTE(random_state=5)),
        ('model', model[1])
    ])
    result = cross_validate(model_pipe, x_selected, y, cv=4, 
                           scoring='f1', return_train_score=True, n_jobs=-1)
    print(model[0])
    print('train_score', result['train_score'].mean()*100)
    print('test_score', result['test_score'].mean()*100)
    print('*'*30)
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
param_grid = {
    'xgboost__max_depth': [3, 4, 5],
    'xgboost__n_estimators': [40, 50,60,70, 100],
    'xgboost__learning_rate': [0.3, 0.4, 0.5]
}

random_pipeline = pip(steps=[
    ('smote', SMOTE(random_state=5)),
    ('xgboost', XGBClassifier())
])

result = RandomizedSearchCV(
    random_pipeline,
    param_grid,
    cv=4,
    return_train_score=True,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

result.fit(x_selected, y)



result.best_score_*100
result.best_params_
result=GridSearchCV(
    random_pipeline,
    param_grid,
    cv=3,
    return_train_score=True,
    scoring='f1',
    n_jobs=-1
)
result.fit(x_selected,y)
result.best_score_*100
result.best_params_
xgboost_pipeline=Pipeline(steps=[('pre processing',pre_pipe),
                                       ('model',XGBClassifier(max_depth=3,n_estimators=100,learning_rate=0.3))])
xgboost_pipeline.fit(x,y)
df.head()
xgboost_pipeline.predict(df.head())
# Model Evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_selected, y, test_size=0.2, random_state=42)

final_model = pip(steps=[
    ('smote', SMOTE(random_state=5)),
    ('xgboost', XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.3))])

final_model.fit(x_train, y_train)
y_pred = final_model.predict(x_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test,y_pred))
import joblib
joblib.dump(final_model,'xgboost.pkl')
joblib.dump(selector,'feature_selector.pkl')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

y_pred_full = final_model.predict(x_test)
print('Accuracy:', accuracy_score(y_test, y_pred_full))
print('Precision:', precision_score(y_test, y_pred_full))
print('Recall:', recall_score(y_test, y_pred_full))
print('F1:', f1_score(y_test, y_pred_full))
## Deployment Using Streamlit 

%%writefile hotel_booking.py
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

st.sidebar.title("🏨 Hotel Booking")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📊 Data Insights", "💡 Analysis Advice", "🔮 Prediction", "📈 Model Performance"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏨 Hotel Booking Cancellation Prediction")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📋 Project Overview")
        st.markdown("""
        This project predicts whether a hotel booking will be **canceled or not**
        using machine learning on real-world hotel booking data.

        The goal is to help hotels identify high-risk bookings early,
        so they can take action to reduce cancellations and revenue loss.
        """)

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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Data Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Insights":
    st.title("📊 Data Insights")
    st.markdown("---")

    # 1 — Cancellation Rate
    st.subheader("1 — Overall Cancellation Rate")
    cancel_rate = df['is_canceled'].value_counts(normalize=True).reset_index()
    cancel_rate.columns = ['Status', 'Rate']
    cancel_rate['Status'] = cancel_rate['Status'].map({0: 'Not Canceled', 1: 'Canceled'})
    cancel_rate['Rate'] = (cancel_rate['Rate'] * 100).round(2)
    fig1 = px.bar(cancel_rate, x='Status', y='Rate', color='Status', text='Rate',
                  color_discrete_sequence=['#2563EB', '#EF4444'])
    fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig1.update_layout(showlegend=False, yaxis_range=[0, 100])
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # 2 — Lead Time vs Cancellation
    st.subheader("2 — Lead Time vs Cancellation")
    fig2 = px.box(df, x='is_canceled', y='lead_time', color='is_canceled',
                  labels={'is_canceled': 'Canceled', 'lead_time': 'Lead Time (days)'},
                  color_discrete_map={0: '#2563EB', 1: '#EF4444'})
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # 3 — Deposit Type vs Cancellation
    st.subheader("3 — Deposit Type vs Cancellation Rate")
    deposit = df.groupby('deposit_type')['is_canceled'].mean().reset_index()
    deposit['is_canceled'] = (deposit['is_canceled'] * 100).round(2)
    deposit.columns = ['Deposit Type', 'Cancellation Rate (%)']
    fig3 = px.bar(deposit, x='Deposit Type', y='Cancellation Rate (%)', color='Deposit Type',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#2563EB', '#F59E0B', '#EF4444'])
    fig3.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig3.update_layout(showlegend=False, yaxis_range=[0, deposit['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # 4 — Market Segment vs Cancellation
    st.subheader("4 — Market Segment vs Cancellation Rate")
    seg = df.groupby('market_segment')['is_canceled'].mean().reset_index()
    seg['is_canceled'] = (seg['is_canceled'] * 100).round(2)
    seg.columns = ['Market Segment', 'Cancellation Rate (%)']
    seg = seg.sort_values('Cancellation Rate (%)', ascending=False)
    fig4 = px.bar(seg, x='Market Segment', y='Cancellation Rate (%)', color='Market Segment',
                  text='Cancellation Rate (%)')
    fig4.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig4.update_layout(showlegend=False, yaxis_range=[0, seg['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # 5 — ADR vs Cancellation
    st.subheader("5 — ADR vs Cancellation")
    fig5 = px.box(df, x='is_canceled', y='adr', color='is_canceled',
                  labels={'is_canceled': 'Canceled', 'adr': 'Average Daily Rate (€)'},
                  color_discrete_map={0: '#2563EB', 1: '#EF4444'})
    fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # 6 — Special Requests vs Cancellation
    st.subheader("6 — Special Requests vs Cancellation Rate")
    special = df.groupby('total_of_special_requests')['is_canceled'].mean().reset_index()
    special['is_canceled'] = (special['is_canceled'] * 100).round(2)
    special.columns = ['Special Requests', 'Cancellation Rate (%)']
    fig6 = px.bar(special, x='Special Requests', y='Cancellation Rate (%)',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#2563EB'])
    fig6.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig6.update_layout(yaxis_range=[0, special['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig6, use_container_width=True)

    st.markdown("---")

    # 7 — Repeated Guest vs Cancellation
    st.subheader("7 — Repeated Guest vs Cancellation Rate")
    repeated = df.groupby('is_repeated_guest')['is_canceled'].mean().reset_index()
    repeated['is_canceled'] = (repeated['is_canceled'] * 100).round(2)
    repeated['is_repeated_guest'] = repeated['is_repeated_guest'].map({0: 'New Guest', 1: 'Repeated Guest'})
    repeated.columns = ['Guest Type', 'Cancellation Rate (%)']
    fig7 = px.bar(repeated, x='Guest Type', y='Cancellation Rate (%)', color='Guest Type',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#2563EB', '#F59E0B'])
    fig7.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig7.update_layout(showlegend=False, yaxis_range=[0, repeated['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig7, use_container_width=True)

    st.markdown("---")

    # 8 — Previous Cancellations vs Cancellation Rate
    st.subheader("8 — Previous Cancellations vs Cancellation Rate")
    prev = df.groupby('previous_cancellations')['is_canceled'].mean().reset_index()
    prev['is_canceled'] = (prev['is_canceled'] * 100).round(2)
    prev.columns = ['Previous Cancellations', 'Cancellation Rate (%)']
    prev = prev[prev['Previous Cancellations'] <= 10]
    fig8 = px.bar(prev, x='Previous Cancellations', y='Cancellation Rate (%)',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#EF4444'])
    fig8.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig8.update_layout(yaxis_range=[0, prev['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig8, use_container_width=True)

    st.markdown("---")

    # 9 — Room Changed vs Cancellation
    st.subheader("9 — Room Changed vs Cancellation Rate")
    room = df.groupby('room_changed')['is_canceled'].mean().reset_index()
    room['is_canceled'] = (room['is_canceled'] * 100).round(2)
    room['room_changed'] = room['room_changed'].map({0: 'Same Room', 1: 'Room Changed'})
    room.columns = ['Room Status', 'Cancellation Rate (%)']
    fig9 = px.bar(room, x='Room Status', y='Cancellation Rate (%)', color='Room Status',
                  text='Cancellation Rate (%)', color_discrete_sequence=['#2563EB', '#F59E0B'])
    fig9.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig9.update_layout(showlegend=False, yaxis_range=[0, room['Cancellation Rate (%)'].max()+10])
    st.plotly_chart(fig9, use_container_width=True)

    st.markdown("---")

    # 10 — Top 10 Countries by Cancellation Rate
    st.subheader("10 — Top 10 Countries by Cancellation Rate")
    country_cancel = df.groupby('country').agg(
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Analysis Advice
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Analysis Advice":
    st.title("💡 Analysis Advice")
    st.markdown("Based on the data analysis, here are key insights and recommendations:")
    st.markdown("---")

    st.subheader("1 — Lead Time")
    st.info("📌 Customers with higher lead times are more likely to cancel their reservations. Hotels should consider sending reminders or offering incentives for bookings made far in advance.")

    st.subheader("2 — Deposit Type")
    st.info("📌 Bookings with Non Refund deposits are rarely canceled. Encouraging non-refundable deposits can significantly reduce cancellation rates.")

    st.subheader("3 — Market Segment")
    st.info("📌 Online TA and Groups segments have the highest cancellation rates. Hotels should apply stricter policies or require deposits for these segments.")

    st.subheader("4 — Special Requests")
    st.info("📌 Customers who make more special requests tend to keep their reservations. Personalized services can increase commitment to bookings.")

    st.subheader("5 — Repeated Guests")
    st.info("📌 Repeated guests have a much lower cancellation rate. Loyalty programs and rewards for returning customers can help reduce overall cancellations.")

    st.subheader("6 — Previous Cancellations")
    st.info("📌 Customers with a history of cancellations are significantly more likely to cancel again. Hotels should flag these customers and require deposits or stricter policies.")

    st.subheader("7 — Room Changed")
    st.info("📌 When the assigned room differs from the reserved room, cancellation patterns change. Ensuring room assignments match reservations can improve customer satisfaction.")

    st.subheader("8 — ADR")
    st.info("📌 Canceled bookings tend to have slightly higher ADR values. Offering flexible pricing or discounts for high-ADR bookings may help retain customers.")

    st.subheader("9 — Country")
    st.info("📌 Some countries have significantly higher cancellation rates. Targeted communication and tailored policies for high-risk countries can be beneficial.")

    st.subheader("10 — General Recommendation")
    st.success("✅ Focus on: Non-refundable deposits, loyalty programs, and personalized communication for high-risk segments to reduce cancellations and improve revenue.")

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
    feature_names = [
        'lead_time', 'agent', 'required_car_parking_spaces',
        'total_of_special_requests', 'room_changed',
        'market_segment_Online TA', 'deposit_type_Non Refund',
        'country (bit 5)', 'country (bit 7)'
    ]
    xgb_model = model.named_steps['xgboost']
    importances = xgb_model.feature_importances_
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    feat_df = feat_df.sort_values('Importance')
    fig_feat = px.bar(feat_df, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale='Blues', text='Importance')
    fig_feat.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig_feat, use_container_width=True)
df[df['adr'] > 1000]

! streamlit run hotel_booking.py
## Best Model

After comparing several machine learning models, XGBoost achieved the best overall performance and was selected as the final model.

## Evaluation Metric

The F1-score was used as the primary evaluation metric because the dataset suffers from class imbalance, making accuracy alone insufficient for evaluating model performance.

## Important Features

Some of the most influential features include:

- Lead Time
- ADR (Average Daily Rate)
- Total Special Requests
- Previous Cancellations
- Deposit Type
- Room Changed
- Market Segment
