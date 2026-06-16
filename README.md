# 📈 Predictive Analytics & Forecasting Dashboard

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![ApexCharts](https://img.shields.io/badge/ApexCharts-00B4D8?style=for-the-badge&logo=chart.js&logoColor=white)

**An interactive machine learning application that forecasts future business trends (e.g. sales, website traffic) using historical time-series data.**  
Includes a premium glassmorphic dark-theme dashboard.

[📂 View Source](https://github.com/adarsh0718/predictive-analytics-forecast)

</div>

---

## 🎬 Overview

This application bridges time-series analysis with standard machine learning. It resamples historical data daily, weekly, or monthly, engineers time-based features (lags, rolling stats, calendar calendars), and trains **Linear Regression**, **Ridge**, **Polynomial**, or **Random Forest** regressors to recursively forecast future values with 95% confidence intervals.

---

## ✨ Features

### 🧠 Machine Learning Engine
- **Multivariate Feature Engineering**: Computes lag features (e.g., $Sales_{t-1}, Sales_{t-7}$), rolling moving averages (7-day/4-week window), and extracts temporal variables (Month, Day of Week, Year, trend indexes).
- **Three Predictive Models**:
  - **Linear Regression**: Ideal for simple linear trends and clear calendar seasonality.
  - **Polynomial Regression**: Adds quadratic interactions ($X^2$) for curves and peak inflection tracking.
  - **Random Forest Regressor**: Ensembled decision trees capable of learning complex non-linear limits and multi-variable dependencies.
- **Recursive Forecasting**: Runs a multi-step recursive loop to project up to 120 days into the future.
- **Confidence Intervals**: Calculates standard residuals variation to project uncertainty boundaries (95% CI) that expand over time.
- **Model Validation**: Chronological train-test split (avoids target leakage) returning MAE, RMSE, $R^2$, and MAPE scores.

### 🎨 Premium UI/UX (Glassmorphic Dark Mode)
- **Interactive Projections Chart**: Plots historical actuals, train/test validation fits, future forecast predictions, and shaded bounds on hover with custom tooltips.
- **Coefficient/Weight Insights**: Dynamically visualizes relative feature impact scores.
- **Residual Errors Histogram**: Evaluates error frequency distributions to inspect if model noise is normally distributed.
- **Interactive Parameter Sidebar**: Adjust split ratios, data granularity (Daily, Weekly, Monthly), and horizon windows instantly.
- **Custom CSV/Excel Uploads**: Fuzzy matches Date and numeric value headers.
- **CSV Data Export**: Downloads projected future coordinates directly.

---

## 📂 Project Structure

```
predictive-analytics-forecast/
│
├── app.py                      # Flask Server (Endpoints: /api/forecast, /api/upload, /api/export)
├── forecasting_engine.py       # ML Pipeline (Feature Lags, Modeling, Evaluation, Projections)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore files
│
├── data/
│   ├── historical_sales.csv    # Default generated historical sales data
│   └── generate_data.py        # Programmatic sales generator script
│
├── templates/
│   └── index.html              # Main dashboard frontend structure
│
└── static/
    ├── css/
    │   └── styles.css          # Premium glassmorphic styling
    └── js/
        ├── app.js              # Orchestrator & API binder
        ├── charts.js           # ApexCharts configurations
        └── tables.js           # Paginated historical table
```

---

## 📋 Custom Data Upload Format

The upload parser auto-detects columns using fuzzy matches. To upload your own dataset, ensure you have:
1. A **Date** column (e.g. `Date`, `Order_Date`, `Timestamp` formatted as YYYY-MM-DD or DD-MM-YYYY).
2. A **Value** column (e.g. `Sales_Revenue`, `Sales`, `Traffic`, `Amount` containing numbers).

---

## 🚀 Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/adarsh0718/predictive-analytics-forecast.git
cd predictive-analytics-forecast
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate data
```bash
python data/generate_data.py
```

### 4. Start Flask server
```bash
python app.py
```
Open **[http://localhost:5001](http://localhost:5001)** in your browser!

---

## 👨‍💻 Author

**Adarsh Peddada**  
Electronics and Computer Engineering Student  
Passionate about Machine Learning, Data Analytics & Software Engineering.

[![GitHub](https://img.shields.io/badge/GitHub-adarsh0718-181717?style=flat-square&logo=github)](https://github.com/adarsh0718)
