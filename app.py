import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")
st.title("📦 E-Commerce Sales Dashboard")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/superstore.csv', encoding='ISO-8859-1')
    return df


df = load_data()

# === Section 1: Raw data
st.write("### 📄 Raw Data Sample")
st.dataframe(df.head())

# === Section 2: Key Metrics
st.write("### 📊 Key Performance Indicators")

total_sales = df['Sales'].sum()
total_profit = df['Profit'].sum()
total_orders = df['Order ID'].nunique()

col1, col2, col3 = st.columns(3)
col1.metric("💰 Total Sales", f"${total_sales:,.2f}")
col2.metric("📈 Total Profit", f"${total_profit:,.2f}")
col3.metric("🧾 Total Orders", total_orders)

# === Section 3: Filter and Sales by Sub-Category
st.write("### 🗂 Category Filter & Sub-Category Sales")
category = st.selectbox("Select Category", df['Category'].unique())

filtered_data = df[df['Category'] == category]

st.write(f"#### 🛍️ Top Products in Category: {category}")
st.dataframe(filtered_data[['Product Name', 'Sales', 'Profit']].sort_values(by="Sales", ascending=False).head())

sales_by_subcat = filtered_data.groupby('Sub-Category')['Sales'].sum().sort_values()

fig1, ax1 = plt.subplots(figsize=(8,5))
sales_by_subcat.plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_title(f'Total Sales by Sub-Category in {category}')
ax1.set_xlabel('Sales')
st.pyplot(fig1)

# === Section 4: Profit by Region
st.write("### 🌍 Profit by Region")

region_profit = df.groupby('Region')['Profit'].sum().sort_values()

fig2, ax2 = plt.subplots(figsize=(8,5))
region_profit.plot(kind='bar', ax=ax2, color='lightgreen')
ax2.set_ylabel('Profit')
ax2.set_title('Total Profit by Region')
st.pyplot(fig2)

# Convert Order Date to datetime if not already
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Date range filter
min_date = df['Order Date'].min()
max_date = df['Order Date'].max()

start_date, end_date = st.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Filter data by date range
mask = (df['Order Date'] >= pd.to_datetime(start_date)) & (df['Order Date'] <= pd.to_datetime(end_date))
filtered_by_date = df[mask]

# Aggregate monthly sales
monthly_sales = (
    filtered_by_date
    .set_index('Order Date')
    .resample('M')['Sales']
    .sum()
)

# Plot sales over time
st.write("### Sales Trend Over Time")
fig3, ax3 = plt.subplots(figsize=(8, 4))
monthly_sales.plot(ax=ax3)
ax3.set_title("Monthly Sales")
ax3.set_xlabel("Month")
ax3.set_ylabel("Sales")
st.pyplot(fig3)

# === Section 5: Customer Segmentation (RFM Analysis)
st.write("### 🧠 Customer Segmentation using RFM Analysis")

# Prepare data
df['Order Date'] = pd.to_datetime(df['Order Date'])
current_date = df['Order Date'].max()

rfm = df.groupby('Customer ID').agg({
    'Order Date': lambda x: (current_date - x.max()).days,
    'Order ID': 'nunique',
    'Sales': 'sum'
}).reset_index()

rfm.columns = ['Customer ID', 'Recency (days)', 'Frequency', 'Monetary']

# Display RFM table
st.write("#### 🧾 RFM Table")
st.dataframe(rfm.sort_values(by='Monetary', ascending=False).head(10))

# Plotting distributions
st.write("#### 📊 Distribution of RFM Metrics")

col1, col2, col3 = st.columns(3)

with col1:
    fig_r, ax_r = plt.subplots()
    sns.histplot(rfm['Recency (days)'], bins=20, ax=ax_r, color='orange')
    ax_r.set_title("Recency Distribution")
    st.pyplot(fig_r)

with col2:
    fig_f, ax_f = plt.subplots()
    sns.histplot(rfm['Frequency'], bins=20, ax=ax_f, color='purple')
    ax_f.set_title("Frequency Distribution")
    st.pyplot(fig_f)

with col3:
    fig_m, ax_m = plt.subplots()
    sns.histplot(rfm['Monetary'], bins=20, ax=ax_m, color='green')
    ax_m.set_title("Monetary Distribution")
    st.pyplot(fig_m)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.write("### 🤖 Customer Segments (K-Means Clustering)")

# Select features for clustering
rfm_features = rfm[['Recency (days)', 'Frequency', 'Monetary']]

# Scale features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# K-Means clustering (choose 4 clusters, you can tune this)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Map cluster number to segment names (you can customize this)
segment_map = {
    0: 'Loyal Customers',
    1: 'At Risk',
    2: 'Champions',
    3: 'Need Attention'
}

rfm['Segment'] = rfm['Cluster'].map(segment_map)

# Show segmented customers count
st.write("#### 🧩 Customer Segments Counts")
st.bar_chart(rfm['Segment'].value_counts())

# Show sample customers per segment
st.write("#### 🧾 Sample Customers by Segment")
segment_selected = st.selectbox("Select Segment to View Customers", rfm['Segment'].unique())
st.dataframe(rfm[rfm['Segment'] == segment_selected].sort_values(by='Monetary', ascending=False).head(10))

from forecasting import preprocess_forecast_data, forecast_sales
import plotly.express as px

# Forecasting
monthly_data = preprocess_forecast_data(df)
forecast = forecast_sales(monthly_data)

# Plot forecast
st.subheader("📈 Sales Forecast (Next 6 Months)")
fig_forecast = px.line(forecast, x='ds', y='yhat', labels={'ds': 'Date', 'yhat': 'Predicted Sales'})
st.plotly_chart(fig_forecast, use_container_width=True)
