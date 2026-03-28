import pandas as pd
import argparse
from sklearn.linear_model import LinearRegression
from datetime import datetime
import warnings


warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="Study Room Occupancy Predictor")
parser.add_argument("--time", type=str, required=True, help="Time to predict for in HH:MM format (e.g., 15:30)")
parser.add_argument("--capacity", type=int, default=20, help="Maximum capacity of the room")
args = parser.parse_args()

def time_to_decimal(timestamp_str):
    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    return dt.hour + (dt.minute / 60.0)

def predict_occupancy(target_time_str, max_capacity):
    log_file = "occupancy_log.csv"
    try:
        df = pd.read_csv(log_file)
    except FileNotFoundError:
        print(f"[ERROR] '{log_file}' not found. Please run tracker.py to gather data first.")
        return

    df['Hour'] = df['Timestamp'].apply(time_to_decimal)
    
    X = df[['Hour']] 
    y = df['Head Count'] 

    model = LinearRegression()
    model.fit(X, y)

    target_dt = datetime.strptime(target_time_str, "%H:%M")
    target_hour = target_dt.hour + (target_dt.minute / 60.0)

    predicted_heads = max(0, int(model.predict([[target_hour]])[0]))
    predicted_rate = (predicted_heads / max_capacity) * 100

    if predicted_rate < 50:
        status = "🟢 LIKELY EMPTY"
    elif predicted_rate < 90:
        status = "🟡 LIKELY BUSY"
    else:
        status = "🔴 LIKELY FULL"

    print("\n" + "="*45)
    print(" 🔮 OCCUPANCY PREDICTION REPORT")
    print("="*45)
    print(f" Data Points Trained : {len(df)} records")
    print(f" Target Time         : {target_time_str}")
    print(f" Predicted Heads     : {predicted_heads}")
    print(f" Room Capacity       : {max_capacity}")
    print(f" Predicted Rate      : {predicted_rate:.1f}%")
    print(f" Forecast Status     : {status}")
    print("="*45 + "\n")

if __name__ == "__main__":
    predict_occupancy(args.time, args.capacity)