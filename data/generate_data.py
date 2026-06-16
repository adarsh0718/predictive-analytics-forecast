import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_historical_data():
    np.random.seed(42)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 6, 15)
    delta = end_date - start_date
    num_days = delta.days + 1
    
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # 1. Base Variables
    baseline_sales = 45000.0
    daily_growth = 12.5 # Slow upward trend
    
    # Pre-select Holiday Dates (approximate Indian/Global major holidays)
    holidays = {
        (1, 26),   # Republic Day
        (8, 15),   # Independence Day
        (10, 2),   # Gandhi Jayanti
        (11, 1),   # Diwali season (varies, but pick fixed dates for simplicity)
        (11, 2),   
        (11, 3),
        (12, 25),  # Christmas
        (1, 1),    # New Year
        (3, 15),   # Holi (approx)
        (10, 24)   # Dussehra (approx)
    }
    
    data = []
    
    for i, date in enumerate(dates):
        # Time features
        day_of_week = date.weekday() # 0 = Monday, 6 = Sunday
        month = date.month
        year = date.year
        
        # Holiday check
        is_holiday = 1 if (month, date.day) in holidays else 0
        
        # 2. Marketing Spend (spikes on weekends and holidays)
        base_ad_spend = np.random.uniform(2000, 8000)
        if day_of_week >= 4: # Fri, Sat, Sun
            base_ad_spend *= 1.4
        if is_holiday:
            base_ad_spend *= 1.8
        marketing_spend = Math_round_or_clamp(base_ad_spend)
        
        # 3. Store Traffic (strongly correlated with marketing spend + random walk)
        traffic_base = 600 + (marketing_spend * 0.12)
        traffic_noise = np.random.normal(0, 80)
        store_traffic = int(max(200, traffic_base + traffic_noise))
        if day_of_week >= 5: # Sat, Sun
            store_traffic = int(store_traffic * 1.25)
        
        # 4. Discount Percentage
        if is_holiday:
            discount = np.random.choice([10, 15, 20, 25])
        elif day_of_week == 5 or day_of_week == 6:
            discount = np.random.choice([0, 5, 10, 15])
        else:
            discount = np.random.choice([0, 0, 5, 5, 10]) # Lower on weekdays
            
        # 5. Seasonality Multipliers
        # Weekly: Higher on Fri (1.15), Sat (1.30), Sun (1.25)
        weekly_mult = 1.0
        if day_of_week == 4: weekly_mult = 1.15
        elif day_of_week == 5: weekly_mult = 1.30
        elif day_of_week == 6: weekly_mult = 1.25
        else: weekly_mult = 0.92
        
        # Monthly: Q4 is high (Nov: 1.35, Dec: 1.45), Q1 slow (Jan/Feb: 0.85)
        monthly_mult = 1.0
        if month == 11: monthly_mult = 1.35
        elif month == 12: monthly_mult = 1.45
        elif month in [1, 2]: monthly_mult = 0.85
        elif month in [5, 6]: monthly_mult = 0.95
        
        # 6. Build Target Revenue
        trend = baseline_sales + (i * daily_growth)
        
        # Combine relationships
        sales = (
            trend + 
            (2.2 * marketing_spend) + 
            (18.5 * store_traffic) + 
            (650 * discount) + 
            (15000 * is_holiday)
        )
        
        # Apply seasonality
        sales *= (weekly_mult * monthly_mult)
        
        # Add random sales noise
        noise = np.random.normal(0, 4500)
        sales_revenue = max(10000, round(sales + noise, 2))
        
        data.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Sales_Revenue": sales_revenue,
            "Marketing_Spend": round(marketing_spend, 2),
            "Store_Traffic": store_traffic,
            "Discount_Percent": discount,
            "Is_Holiday": is_holiday
        })
        
    df = pd.DataFrame(data)
    df.to_csv("data/historical_sales.csv", index=False)
    print(f"Generated {len(df)} daily records in data/historical_sales.csv")

def Math_round_or_clamp(val):
    return round(max(0, val), 2)

if __name__ == "__main__":
    generate_historical_data()
