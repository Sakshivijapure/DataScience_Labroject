# eda_code.py

# -----------------------------------------------------------
# NIFTY 15 STOCK ANALYSIS & PREDICTION - GROUP ARAMBOL
# -----------------------------------------------------------

# Members:
# 1. Data Collection & Preprocessing
# 2. Exploratory Data Analysis
# 3. Model Building & Evaluation
# 4. Interactive Dashboard & Insights

# -----------------------------------------------------------
# ---- IMPORT LIBRARIES ---- #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st
import sqlite3
import os
from datetime import date

# ---- PAGE CONFIG ---- #
st.set_page_config(page_title="NIFTY 15 Stock Dashboard", layout="wide", page_icon="📈")

# ---- CUSTOM CSS ---- #
st.markdown(
    """
    <style>
    /* Main background and font */
    .main { background-color: #f4f6f7; font-family: 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: #1f2937; }

    /* Sidebar background and text */
    [data-testid="stSidebar"] {
        background-color: #1f2937;
        color: white;
    }
    [data-testid="stSidebar"] .stRadio { color: white; }

    .css-18e3th9 {padding-top: 2rem;}
    </style>
    """, unsafe_allow_html=True
)

# ---- SIDEBAR NAVIGATION ---- #
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to Section:",
    [
        "1. Data Collection & Cleaning",
        "2. Exploratory Data Analysis",
        "3. Modeling & Evaluation",
        "4. Interactive Dashboard",
    ],
)

# ---- LOAD DATA ---- #
file_path = "cleaned_nifty15.csv"
if not os.path.exists(file_path):
    st.error("File 'cleaned_nifty15.csv' not found. Please add it and rerun.")
    st.stop()

data = pd.read_csv(file_path)
data.dropna(inplace=True)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values(by=["Symbol", "Date"]).drop_duplicates()

# ---- FEATURE ENGINEERING ---- #
data["Daily_Return"] = data.groupby("Symbol")["Close"].pct_change()
data["MA_20"] = data.groupby("Symbol")["Close"].transform(lambda x: x.rolling(20).mean())
data["MA_50"] = data.groupby("Symbol")["Close"].transform(lambda x: x.rolling(50).mean())
data["Volatility"] = data.groupby("Symbol")["Daily_Return"].transform(lambda x: x.rolling(20).std())

# ---- SAVE TO DATABASE ---- #
conn = sqlite3.connect("stocks.db")
data.to_sql("stock_data", conn, if_exists="replace", index=False)
conn.close()

# ===========================================================
# 1. DATA COLLECTION & CLEANING
# ===========================================================
if section == "1. Data Collection & Cleaning":
    st.title("Data Collection & Cleaning")
    st.subheader("Preview of Raw Data")
    st.dataframe(data.head())

    st.subheader("Data Cleaning Summary")
    st.markdown("""
    - Missing values removed
    - Duplicates dropped
    - Dates converted to proper datetime format
    - Data sorted by Symbol and Date
    """)
    st.success("Data cleaned and stored in SQLite database (stocks.db)")

    st.subheader("New Features Added")
    st.markdown("Daily_Return, MA_20, MA_50, and Volatility have been engineered for further analysis.")
    st.dataframe(data[["Symbol", "Date", "Close", "Daily_Return", "MA_20", "MA_50", "Volatility"]].tail())

    st.subheader("Descriptive Statistics")
    st.dataframe(data.describe().T.style.background_gradient(cmap="PuBu"))

# ===========================================================
# 2. EXPLORATORY DATA ANALYSIS
# ===========================================================
elif section == "2. Exploratory Data Analysis":
    st.title("Exploratory Data Analysis (EDA)")
    st.markdown("Visualize trends, returns, and correlations interactively.")

    st.subheader("Stock Price Trends Over Time")
    fig1 = px.line(
        data,
        x="Date",
        y="Close",
        color="Symbol",
        title="Stock Price Comparison",
        color_discrete_sequence=px.colors.qualitative.Set2,
        template="plotly_dark",
        hover_data={"Volume": True},
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Top Performing Stocks")
    data["Cumulative_Return"] = data.groupby("Symbol")["Daily_Return"].transform(lambda x: (1 + x).cumprod() - 1)
    final_returns = data.groupby("Symbol")["Cumulative_Return"].last().sort_values(ascending=False)
    top5 = final_returns.head(5).reset_index()
    fig2 = px.bar(
        top5,
        x="Symbol",
        y="Cumulative_Return",
        color="Cumulative_Return",
        color_continuous_scale="Viridis",
        title="Top 5 Stocks by Cumulative Return",
        template="plotly_white",
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Heatmap")
    pivot_data = data.pivot_table(values="Close", index="Date", columns="Symbol")
    corr = pivot_data.corr()
    fig3, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig3)

# ===========================================================
# 3. MODELING & EVALUATION
# ===========================================================
elif section == "3. Modeling & Evaluation":
    st.title("Predictive Modeling (Linear Regression)")

    # Prepare the data
    model_data = data.dropna(subset=["MA_20", "MA_50", "Volume"])
    X = model_data[["MA_20", "MA_50", "Volume"]]
    y = model_data["Close"]

    # Split into train and test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    # Build and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(((y_test - predictions) ** 2).mean())
    r2 = r2_score(y_test, predictions)

    # Display metrics
    st.subheader("Model Evaluation Metrics")
    st.write("Mean Absolute Error (MAE):", round(mae, 3))
    st.write("Root Mean Squared Error (RMSE):", round(rmse, 3))
    st.write("R² Score:", round(r2, 3))

    # Scatter Plot: Actual vs Predicted
    st.subheader("Actual vs Predicted Scatter Plot")
    fig1 = px.scatter(
        x=y_test,
        y=predictions,
        labels={"x": "Actual Prices", "y": "Predicted Prices"},
        title="Actual vs Predicted Prices",
        template="plotly_white",
        size_max=15
    )
    fig1.add_shape(
        type="line",
        x0=min(y_test),
        y0=min(y_test),
        x1=max(y_test),
        y1=max(y_test),
        line=dict(color="red", width=2, dash="dash")
    )
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Blue points = predicted vs actual; Red dashed line = perfect predictions.")

    # Line Plot: Actual vs Predicted (first 100 points)
    st.subheader("Comparison of Actual vs Predicted (First 100 Points)")
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.values[:100], label="Actual", color="green", linewidth=2)
    plt.plot(predictions[:100], label="Predicted", color="red", linestyle="--", linewidth=2)
    plt.title("Actual vs Predicted Closing Prices (Sample of 100)")
    plt.xlabel("Records")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(plt)
    st.caption("If both lines overlap closely, the model predicts well.")

# ===========================================================
# 4. INTERACTIVE DASHBOARD
# ===========================================================
elif section == "4. Interactive Dashboard":
    st.title("Interactive Dashboard")

    symbols = st.multiselect(
        "Select Stock(s) to Compare",
        data["Symbol"].unique(),
        default=data["Symbol"].unique()[:3],
    )
    filtered_data = data[data["Symbol"].isin(symbols)]

    st.subheader("Price Trend")
    fig_trend = px.line(
        filtered_data, x="Date", y="Close", color="Symbol",
        title="Stock Price Trend",
        template="plotly_dark"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Growth Over Period")
    min_date, max_date = data["Date"].min().date(), data["Date"].max().date()
    date_range = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )
    growth_df = data[(data["Date"].dt.date >= date_range[0]) & (data["Date"].dt.date <= date_range[1])]
    growth_summary = growth_df.groupby("Symbol")["Close"].agg(["first", "last"])
    growth_summary["% Growth"] = ((growth_summary["last"] - growth_summary["first"]) / growth_summary["first"]) * 100
    growth_summary = growth_summary.sort_values("% Growth", ascending=False).reset_index()

    fig_growth = px.bar(
        growth_summary, x="Symbol", y="% Growth",
        color="% Growth", color_continuous_scale="Viridis",
        title="Company Growth in Selected Period",
        template="plotly_white"
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    st.subheader("Volatility Comparison")
    vol_df = data.groupby("Symbol")["Volatility"].mean().reset_index().sort_values("Volatility", ascending=False)
    fig_vol = px.bar(
        vol_df, x="Symbol", y="Volatility", color="Volatility",
        title="Average Volatility by Stock",
        template="plotly_white"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # Summary Metrics
    st.subheader("Summary Metrics")
    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div style="background-color:#ffffff; padding:20px; border-radius:12px; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h4 style="color:#111827; margin:0;">Total Stocks</h4>
        <h2 style="color:#111827; margin:0;">{len(data['Symbol'].unique())}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div style="background-color:#ffffff; padding:20px; border-radius:12px; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h4 style="color:#111827; margin:0;">Data Range</h4>
        <h2 style="color:#111827; margin:0;">{min_date} → {max_date}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div style="background-color:#ffffff; padding:20px; border-radius:12px; text-align:center; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
        <h4 style="color:#111827; margin:0;">Total Records</h4>
        <h2 style="color:#111827; margin:0;">{len(data)}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.success("Dashboard ready!")
    st.caption("Developed by Group Arambol | Tools: Streamlit, Plotly, Scikit-learn")
