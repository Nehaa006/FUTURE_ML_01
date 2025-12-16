import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# --- 1. LOAD DATA ---
# We try different encodings because Superstore datasets often have special characters
try:
    df = pd.read_csv('superstore.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('superstore.csv', encoding='windows-1252')

print("Data Loaded Successfully!")
print(df.head())

# --- 2. PREPARE DATA ---
# Prophet requires two specific columns:
# 'ds' = The Date
# 'y'  = The value we want to predict (Sales)

# Check if 'Order Date' exists, otherwise try to find the date column
if 'Order Date' in df.columns:
    date_col = 'Order Date'
else:
    # Auto-detect the first column with 'Date' in the name
    date_col = [col for col in df.columns if 'Date' in col][0]

# Ensure it's in datetime format
df[date_col] = pd.to_datetime(df[date_col])

# Group by Date and Sum Sales (Daily Sales)
# We aggregate because we want the TOTAL sales per day, not individual transaction rows
daily_sales = df.groupby(date_col)['Sales'].sum().reset_index()

# Rename columns for Prophet
daily_sales.columns = ['ds', 'y']

# --- 3. TRAIN THE MODEL ---
print("Training the AI model... (This might take a moment)")
m = Prophet(interval_width=0.95, daily_seasonality=False)
m.fit(daily_sales)

# --- 4. PREDICT THE FUTURE ---
# Create a placeholder for the next 90 days
future = m.make_future_dataframe(periods=90) 
forecast = m.predict(future)

# --- 5. VISUALIZE & SAVE ---
# Plot the forecast
plt.figure(figsize=(10, 6))
m.plot(forecast)
plt.title("Sales Forecast: Next 90 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Save the detailed forecast to CSV for Power BI
# We only keep the essential columns for the dashboard
output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
output.to_csv('forecast_for_dashboard.csv', index=False)

print("✅ Success! Predictions saved to 'forecast_for_dashboard.csv'")
print("Now you can load this file into Power BI.")

#pip install pandas fbprophet matplotlib scikit-learn