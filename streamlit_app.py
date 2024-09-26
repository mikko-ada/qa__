import openai
import streamlit as st
from io import StringIO
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import math
import os

# Access the API key securely
openai.api_key = st.secrets["openai_api_key"]

if not openai.api_key:
    st.error("OpenAI API key not found. Please set it in your Streamlit secrets.")
    st.stop()

def rename_dataset_columns(dataframe):
    if dataframe.empty:
        st.error("No data to process.")
        return dataframe
    dataframe.columns = [re.sub(r'[%@#&$()]+', '', col).replace(' ', '_').strip('_') for col in dataframe.columns]
    return dataframe

def convert_datatype(df):
    for c in df.columns[df.dtypes == 'object']:
        try:
            df[c] = pd.to_datetime(df[c], errors='raise')
        except Exception as e:
            st.error(f"Error converting column {c}: {str(e)}")
    df = df.convert_dtypes()
    return df

def load_data(files):
    data_frames = []
    for uploaded_file in files:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        try:
            data_frame = pd.read_csv(stringio)
            data_frames.append(data_frame)
        except pd.errors.EmptyDataError:
            st.error(f"Empty data in file: {uploaded_file.name}")
            continue
    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        return pd.DataFrame()

uploaded_files = st.file_uploader("Choose CSV files", accept_multiple_files=True)
orders_df = pd.DataFrame()
if uploaded_files:
    orders_df = load_data(uploaded_files)
    if not orders_df.empty:
        orders_df = rename_dataset_columns(orders_df)
        orders_df = convert_datatype(orders_df)

if not orders_df.empty:
    # Setup for RFM analysis
    st.title("RFM Analysis Configuration")
    date_col = st.selectbox("Select Date Column", orders_df.columns, index=orders_df.columns.get_loc("date") if "date" in orders_df.columns else 0)
    customer_id_col = st.selectbox("Select Customer ID Column", orders_df.columns, index=orders_df.columns.get_loc("customer_id") if "customer_id" in orders_df.columns else 0)
    monetary_value_col = st.selectbox("Select Monetary Value Column", orders_df.columns, index=orders_df.columns.get_loc("total_spent") if "total_spent" in orders_df.columns else 0)

    # Function to perform RFM analysis
    def rfm_analysis(df, date_column, customer_id_column, monetary_value_column):
        if df.empty:
            st.error("Data is not loaded or is incorrect.")
            return None
        df = df.dropna(subset=[date_column, customer_id_column, monetary_value_column])
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        current_date = df[date_column].max()
        rfm = df.groupby(customer_id_column).agg({
            date_column: lambda x: (current_date - x.max()).days,
            customer_id_column: 'count',
            monetary_value_column: 'sum'
        }).rename(columns={
            date_column: 'Recency',
            customer_id_column: 'Frequency',
            monetary_value_column: 'MonetaryValue'
        })
        return rfm

    if st.button("Run RFM Analysis"):
        rfm_df = rfm_analysis(orders_df, date_col, customer_id_col, monetary_value_col)
        if rfm_df is not None:
            st.write("RFM Analysis Results", rfm_df)
            fig = px.bar(rfm_df, x=rfm_df.index, y='Frequency', title="Customer Purchase Frequency")
            st.plotly_chart(fig, use_container_width=True)

            # Additional visualizations based on RFM results
            recency_fig = px.histogram(rfm_df, x="Recency", title="Recency Distribution")
            monetary_fig = px.histogram(rfm_df, x="MonetaryValue", title="Monetary Value Distribution")
            st.plotly_chart(recency_fig, use_container_width=True)
            st.plotly_chart(monetary_fig, use_container_width=True)
