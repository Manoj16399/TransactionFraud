# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(layout="wide", page_title="Financial Fraud Detection Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Synthetic_Financial_Fraud_Dataset__10k_.csv")
    df.columns = df.columns.str.strip()  # Strip any extra spaces from column names
    return df

df = load_data()

# Title and Description
st.title("\U0001F4B0 Financial Fraud Detection Dashboard")
st.markdown("""
This dashboard helps stakeholders and the Financial Security Director to analyze transaction data
and detect fraudulent activity through macro and micro visual insights.
""")

# Tabs for organization
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Temporal Trends", "Transaction Features", "Account Insights", "Predictive Analysis"])

# Tab 1: Overview
with tab1:
    st.header("\U0001F4C8 Dataset Overview")
    st.markdown("Basic metrics and dataset summary to understand transaction scale and fraud ratio.")

    col1, col2 = st.columns(2)
    col1.metric("Total Transactions", len(df))

    if 'isFraud' in df.columns:
        col2.metric("Total Fraudulent Transactions", int(df['isFraud'].sum()))
    else:
        col2.warning("'isFraud' column not found in dataset")

    st.markdown("### Sample Data")
    st.dataframe(df.head(10))

    st.markdown("### Fraud vs Non-Fraud Distribution")
    if 'isFraud' in df.columns:
        fraud_counts = df['isFraud'].value_counts()
        fig1, ax1 = plt.subplots()
        ax1.pie(fraud_counts, labels=["Non-Fraud", "Fraud"], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
        ax1.axis('equal')
        st.pyplot(fig1)
    else:
        st.warning("Cannot generate fraud distribution without 'isFraud' column.")

# Tab 2: Temporal Trends
with tab2:
    st.header("\U0001F4C5 Temporal Trends")
    st.markdown("Analysis of fraud patterns over transaction steps (time-based simulation).")

    if 'step' in df.columns and 'isFraud' in df.columns:
        fraud_trend = df.groupby('step')['isFraud'].mean().reset_index()
        fig2 = px.line(fraud_trend, x='step', y='isFraud',
                       labels={'isFraud': 'Fraud Rate', 'step': 'Time Step'},
                       title='Fraud Rate Over Time')
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Required columns 'step' and/or 'isFraud' not found.")

# Tab 3: Transaction Features
with tab3:
    st.header("\U0001F4B3 Transaction Feature Analysis")
    st.markdown("Explore how transaction amount and type influence fraud likelihood.")

    if 'isFraud' in df.columns and 'amount' in df.columns:
        st.markdown("### Amount Distribution by Fraud Status")
        fig3, ax3 = plt.subplots()
        sns.histplot(data=df, x='amount', hue='isFraud', kde=True, bins=50, ax=ax3)
        st.pyplot(fig3)
    else:
        st.warning("Cannot plot amount distribution without 'amount' and 'isFraud'.")

    if 'type' in df.columns and 'isFraud' in df.columns:
        st.markdown("### Transaction Type vs Fraud")
        fraud_by_type = df.groupby('type')['isFraud'].mean().reset_index()
        fig4 = px.bar(fraud_by_type, x='type', y='isFraud',
                      labels={'isFraud': 'Fraud Rate'},
                      title='Fraud Rate by Transaction Type')
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Required columns 'type' and/or 'isFraud' not found.")

# Tab 4: Account/User Analysis
with tab4:
    st.header("\U0001F465 Account Activity Insights")
    st.markdown("Explore accounts with most activity and fraud incidence.")

    if 'nameOrig' in df.columns and 'isFraud' in df.columns:
        top_accounts = df[df['isFraud'] == 1]['nameOrig'].value_counts().head(10)
        st.bar_chart(top_accounts)
        st.markdown("Top 10 Accounts Involved in Fraudulent Transactions")

    if 'nameDest' in df.columns and 'isFraud' in df.columns:
        st.markdown("Destination Account Fraud Frequency")
        dest_fraud = df[df['isFraud'] == 1]['nameDest'].value_counts().head(10)
        st.bar_chart(dest_fraud)

# Tab 5: Predictive Feature Overview
with tab5:
    st.header("\U0001F52C Feature Importance Insight")
    st.markdown("Using basic statistical correlation to highlight important predictors.")

    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[num_cols].corr()
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax5)
    st.pyplot(fig5)

    st.markdown("""
    The heatmap shows correlation between numerical features and fraud status. Values closer to 1 or -1 indicate stronger relationships.
    """)
