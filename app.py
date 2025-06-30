import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Financial Fraud Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Synthetic_Financial_Fraud_Dataset__10k_.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filter Data")
amount_range = st.sidebar.slider("Transaction Amount Range", 0, int(df["Amount"].max()), (0, 5000))
transaction_types = st.sidebar.multiselect("Select Transaction Types", df["Transaction Type"].unique(), df["Transaction Type"].unique())
risk_filter = st.sidebar.slider("Risk Score Range", 0, int(df["Risk Score"].max()), (0, 100))

filtered_df = df[
    (df["Amount"] >= amount_range[0]) &
    (df["Amount"] <= amount_range[1]) &
    (df["Transaction Type"].isin(transaction_types)) &
    (df["Risk Score"] >= risk_filter[0]) &
    (df["Risk Score"] <= risk_filter[1])
]

# Title
st.title("💳 Financial Fraud Detection Dashboard")
st.markdown("A go-to platform for Directors and Stakeholders to monitor, analyze, and understand fraud dynamics at macro and micro levels.")

# Dataset preview
st.subheader("📁 Data Sample")
st.dataframe(filtered_df.head(10))

# Tab Layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Risk & Trust", "🕒 Temporal Trends", "🌍 Geography & Device"])

with tab1:
    st.subheader("1. Fraud vs Non-Fraud Counts")
    st.markdown("Shows the number of fraudulent and legitimate transactions.")
    fig = px.histogram(df, x="Is Fraud", color="Is Fraud", color_discrete_sequence=["green", "red"])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2. Transaction Type vs Fraud")
    st.markdown("Breakdown of fraud status across transaction types.")
    fig = px.histogram(filtered_df, x="Transaction Type", color="Is Fraud", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3. Fraud by Device Type")
    fig = px.histogram(filtered_df, x="Device Type", color="Is Fraud", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("4. Fraud Rate by Country")
    fraud_country = df.groupby("Country")["Is Fraud"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(fraud_country)

with tab2:
    st.subheader("5. Distribution of Risk Score")
    fig, ax = plt.subplots()
    sns.histplot(filtered_df["Risk Score"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("6. Device Trust Score by Fraud Status")
    fig = px.box(filtered_df, x="Is Fraud", y="Device Trust Score", color="Is Fraud")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("7. High Risk Country vs Fraud")
    fig = px.histogram(filtered_df, x="Is High Risk Country", color="Is Fraud", barmode="group")
    st.plotly_chart(fig)

    st.subheader("8. Previous Fraud History vs Current Fraud")
    fig = px.histogram(filtered_df, x="Has Fraud History", color="Is Fraud", barmode="group")
    st.plotly_chart(fig)

with tab3:
    st.subheader("9. Fraud Over Time of Day")
    fig = px.histogram(filtered_df, x="Time", color="Is Fraud", nbins=24)
    st.plotly_chart(fig)

    st.subheader("10. Time Since Last Transaction vs Fraud")
    fig = px.scatter(filtered_df, x="Time since last transaction", y="Amount", color="Is Fraud")
    st.plotly_chart(fig)

    st.subheader("11. Fraud by User Tenure")
    fig = px.box(filtered_df, x="Is Fraud", y="User Tenure")
    st.plotly_chart(fig)

    st.subheader("12. Age Distribution of Users")
    fig = px.histogram(filtered_df, x="User Age", color="Is Fraud")
    st.plotly_chart(fig)

with tab4:
    st.subheader("13. Geographic Distribution of Fraud")
    fraud_geo = df[df["Is Fraud"] == 1]["Region"].value_counts().head(10)
    st.bar_chart(fraud_geo)

    st.subheader("14. IP Address Heatmap by Fraud")
    fraud_ip = df.groupby("Ip_Address")["Is Fraud"].sum().sort_values(ascending=False).head(10)
    st.bar_chart(fraud_ip)

    st.subheader("15. Foreign Transactions vs Fraud")
    fig = px.histogram(filtered_df, x="Is Foreign Transaction?", color="Is Fraud", barmode="group")
    st.plotly_chart(fig)

    st.subheader("16. Country vs Risk Score")
    fig = px.box(filtered_df, x="Country", y="Risk Score", color="Is Fraud")
    st.plotly_chart(fig)

# Summary Section
st.subheader("📌 Summary Stats")
col1, col2, col3 = st.columns(3)
col1.metric("Total Transactions", len(df))
col2.metric("Total Fraud Cases", int(df["Is Fraud"].sum()))
col3.metric("Fraud Rate", f"{100 * df['Is Fraud'].mean():.2f}%")
