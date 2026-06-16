import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def preprocess_and_forecast(filepath, model_type='linear', horizon=30, split_ratio=0.8, granularity='D'):
    """
    Loads historical sales data, aggregates it, runs chronological train/test split,
    creates time-series lag/rolling features, trains the selected model, computes metrics,
    and runs recursive forecasting for the future horizon.
    """
    # 1. Load and aggregate data
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Identify target and auxiliary variables
    # Schema check and fuzzy matching for target variable
    target_col = None
    for col in df.columns:
        if col.lower() in ['sales_revenue', 'sales', 'revenue', 'value']:
            target_col = col
            break
    if not target_col:
        target_col = df.columns[1] # fallback to second column
        
    # Aggregate data if granularity is different
    df.set_index('Date', inplace=True)
    if granularity == 'W':
        df_resampled = df.resample('W').agg({
            target_col: 'sum',
            'Marketing_Spend': 'sum' if 'Marketing_Spend' in df.columns else 'first',
            'Store_Traffic': 'sum' if 'Store_Traffic' in df.columns else 'first',
            'Discount_Percent': 'mean' if 'Discount_Percent' in df.columns else 'first',
            'Is_Holiday': 'sum' if 'Is_Holiday' in df.columns else 'first'
        })
    elif granularity == 'M':
        df_resampled = df.resample('ME').agg({
            target_col: 'sum',
            'Marketing_Spend': 'sum' if 'Marketing_Spend' in df.columns else 'first',
            'Store_Traffic': 'sum' if 'Store_Traffic' in df.columns else 'first',
            'Discount_Percent': 'mean' if 'Discount_Percent' in df.columns else 'first',
            'Is_Holiday': 'sum' if 'Is_Holiday' in df.columns else 'first'
        })
    else:
        df_resampled = df.copy()
        
    df_resampled.reset_index(inplace=True)
    df_resampled = df_resampled.dropna(subset=[target_col])
    
    # 2. Feature Engineering for Time Series
    # Calendar features
    df_resampled['Year'] = df_resampled['Date'].dt.year
    df_resampled['Month'] = df_resampled['Date'].dt.month
    df_resampled['Day'] = df_resampled['Date'].dt.day
    df_resampled['DayOfWeek'] = df_resampled['Date'].dt.dayofweek
    df_resampled['TimeIndex'] = np.arange(len(df_resampled)) # Linear trend feature
    
    # Lag and Rolling features (adjust window size based on granularity)
    lag_days = [1, 2, 7] if granularity == 'D' else [1, 2, 4]
    roll_window = 7 if granularity == 'D' else 4
    
    for lag in lag_days:
        df_resampled[f'Lag_{lag}'] = df_resampled[target_col].shift(lag)
        
    df_resampled['Roll_Mean'] = df_resampled[target_col].shift(1).rolling(window=roll_window).mean()
    df_resampled['Roll_Std'] = df_resampled[target_col].shift(1).rolling(window=roll_window).std()
    
    # Drop rows with NaN (from shifting/rolling)
    df_features = df_resampled.dropna().copy().reset_index(drop=True)
    
    if len(df_features) < 15:
        raise ValueError("Insufficient data points for forecasting. Please upload a larger dataset.")
        
    # Feature list for modeling
    feature_cols = ['TimeIndex', 'Month', 'DayOfWeek'] + [f'Lag_{l}' for l in lag_days] + ['Roll_Mean']
    
    # Chronological Train-Test Split (No random shuffling in time series!)
    split_idx = int(len(df_features) * split_ratio)
    train_df = df_features.iloc[:split_idx]
    test_df = df_features.iloc[split_idx:]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    # 3. Model Training
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    elif model_type == 'polynomial':
        model = make_pipeline(PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0))
    else:
        model = LinearRegression() # default
        
    model.fit(X_train, y_train)
    
    # 4. Evaluation
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
    
    # Residuals
    residuals = y_train - y_pred_train
    std_residual = np.std(residuals)
    
    # Feature Importances/Coefficients
    feature_importance = {}
    if model_type == 'random_forest':
        importances = model.feature_importances_
        for col, imp in zip(feature_cols, importances):
            feature_importance[col] = float(imp)
    elif model_type == 'linear':
        coefs = model.coef_
        max_coef = max(np.abs(coefs)) if len(coefs) > 0 else 1.0
        for col, coef in zip(feature_cols, coefs):
            # Normalize for simple importance visualization
            feature_importance[col] = float(np.abs(coef) / max_coef)
    else:
        # For polynomial, extract ridge coefficients
        coefs = model.steps[1][1].coef_
        max_coef = max(np.abs(coefs)) if len(coefs) > 0 else 1.0
        # Summarize polynomial importances by base features
        for i, col in enumerate(feature_cols):
            feature_importance[col] = float(np.abs(coefs[i]) / max_coef) if i < len(coefs) else 0.1
            
    # 5. Future Recursive Forecasting
    future_forecast = []
    future_dates = []
    
    # Start recursive forecast from the last record of our processed dataset
    last_idx = len(df_features) - 1
    last_row = df_features.iloc[last_idx]
    
    # Initialize rolling history for recursive lags
    history_sales = df_features[target_col].tolist()
    last_date = df_features['Date'].iloc[last_idx]
    
    freq_map = {'D': 'D', 'W': 'W', 'M': 'M'}
    date_offsets = pd.date_range(start=last_date, periods=horizon + 1, freq=freq_map[granularity])[1:]
    
    time_idx = last_row['TimeIndex']
    
    for step in range(horizon):
        next_date = date_offsets[step]
        time_idx += 1
        
        # Build features for this step
        step_features = {}
        step_features['TimeIndex'] = time_idx
        step_features['Month'] = next_date.month
        step_features['DayOfWeek'] = next_date.dayofweek
        
        # Lag features from history_sales
        for lag in lag_days:
            step_features[f'Lag_{lag}'] = history_sales[-lag]
            
        # Rolling mean of the last 'roll_window' sales
        step_features['Roll_Mean'] = np.mean(history_sales[-roll_window:])
        
        # Convert to model input array matching feature_cols order
        input_vector = [step_features[col] for col in feature_cols]
        input_df = pd.DataFrame([input_vector], columns=feature_cols)
        
        # Predict
        pred_val = float(model.predict(input_df)[0])
        pred_val = max(0.0, pred_val) # Clamp to zero
        
        # Append to history and forecast lists
        history_sales.append(pred_val)
        
        # Confidence intervals (uncertainty grows with step index)
        # CI width = 1.96 * std_residual * sqrt(step_index + 1)
        ci_width = 1.96 * std_residual * np.sqrt(step + 1)
        
        future_forecast.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "prediction": round(pred_val, 2),
            "lower_bound": round(max(0.0, pred_val - ci_width), 2),
            "upper_bound": round(pred_val + ci_width, 2)
        })
        
    # Format actual history + validation test projections for chart plotting
    historical_points = []
    for _, row in df_resampled.iterrows():
        historical_points.append({
            "date": row['Date'].strftime("%Y-%m-%d"),
            "actual": float(row[target_col]),
            "predicted": None,
            "type": "history"
        })
        
    # Validation predictions overlap
    test_points = []
    for idx, row in test_df.iterrows():
        test_points.append({
            "date": row['Date'].strftime("%Y-%m-%d"),
            "actual": float(row[target_col]),
            "predicted": float(y_pred_test[idx - split_idx]),
            "type": "validation"
        })
        
    return {
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mape": round(mape, 2)
        },
        "feature_importance": feature_importance,
        "historical_data": historical_points,
        "validation_data": test_points,
        "forecast_data": future_forecast,
        "residuals": residuals.tolist()
    }
