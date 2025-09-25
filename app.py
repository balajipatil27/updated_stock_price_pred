import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file, jsonify
import datetime as dt
import yfinance as yf
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os
plt.style.use("fivethirtyeight")

app = Flask(__name__)
CORS(app)
# Create static directory if it doesn't exist
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Global variables to store model and scaler
model = None
feature_scaler = None
target_scaler = None
current_stock_data = None

def prepare_model_data(df):
    """Prepare features and target for model training"""
    # Create features: Open, High, Low, Volume, and some technical indicators
    features = pd.DataFrame()
    features['Open'] = df['Open']
    features['High'] = df['High']
    features['Low'] = df['Low']
    features['Volume'] = df['Volume']
    
    # Add technical indicators as features
    features['Price_Range'] = df['High'] - df['Low']
    features['Mid_Price'] = (df['High'] + df['Low']) / 2
    features['Volume_MA'] = df['Volume'].rolling(window=5).mean()
    features['Price_MA'] = df['Close'].rolling(window=5).mean()
    
    # Target is the closing price
    target = df['Close']
    
    # Remove rows with NaN values
    combined = pd.concat([features, target], axis=1).dropna()
    features_clean = combined.iloc[:, :-1]
    target_clean = combined.iloc[:, -1]
    
    return features_clean, target_clean

def train_prediction_model(df):
    """Train a model to predict closing price based on OHLV data"""
    global model, feature_scaler, target_scaler
    
    features, target = prepare_model_data(df)
    
    if len(features) < 10:
        return False
    
    # Scale features and target
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    features_scaled = feature_scaler.fit_transform(features)
    target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
    
    # Train Random Forest model (better for this type of prediction)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(features_scaled, target_scaled)
    
    return True

def predict_closing_price(open_price, high_price, low_price, volume):
    """Predict closing price based on OHLV inputs"""
    global model, feature_scaler, target_scaler, current_stock_data
    
    if model is None or feature_scaler is None or target_scaler is None:
        return None, "Model not trained"
    
    try:
        # Create feature vector
        price_range = high_price - low_price
        mid_price = (high_price + low_price) / 2
        
        # Use recent averages for MA features (or use provided values if available)
        if current_stock_data is not None and len(current_stock_data) > 5:
            volume_ma = current_stock_data['Volume'].tail(5).mean()
            price_ma = current_stock_data['Close'].tail(5).mean()
        else:
            # Fallback to simple estimates
            volume_ma = volume
            price_ma = mid_price
        
        features = np.array([[open_price, high_price, low_price, volume, 
                             price_range, mid_price, volume_ma, price_ma]])
        
        # Scale features
        features_scaled = feature_scaler.transform(features)
        
        # Predict
        prediction_scaled = model.predict(features_scaled)[0]
        
        # Inverse transform to get actual price
        prediction = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
        
        return prediction, None
    
    except Exception as e:
        return None, str(e)

@app.route('/', methods=['GET', 'POST'])
def index():
    global current_stock_data
    
    if request.method == 'POST':
        stock = request.form.get('stock')
        if not stock:
            stock = 'AAPL'
        
        try:
            # Define the start and end dates for stock data
            start = dt.datetime(2020, 1, 1)
            end = dt.datetime(2024, 10, 1)
            
            # Download stock data
            print(f"Downloading data for {stock}...")
            df = yf.download(stock, start=start, end=end)
            
            if df.empty:
                return render_template('index.html', error=f"No data found for stock symbol: {stock}")
            
            print(f"Data downloaded successfully. Shape: {df.shape}")
            
            # Handle multi-level columns if they exist
            if hasattr(df.columns, 'nlevels') and df.columns.nlevels > 1:
                df.columns = df.columns.droplevel(1)
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                return render_template('index.html', error=f"Missing required data columns: {missing_columns}")
            
            # Store current stock data globally
            current_stock_data = df.copy()
            
            # Train the prediction model
            model_trained = train_prediction_model(df)
            if not model_trained:
                return render_template('index.html', error="Could not train prediction model - insufficient data")
            
            # Descriptive Data
            data_desc = df.describe()
            
            # Exponential Moving Averages
            ema20 = df['Close'].ewm(span=20, adjust=False).mean()
            ema50 = df['Close'].ewm(span=50, adjust=False).mean()
            ema100 = df['Close'].ewm(span=100, adjust=False).mean()
            ema200 = df['Close'].ewm(span=200, adjust=False).mean()
            
            # Create visualizations (keeping your existing plotting code)
            # Plot 1: Closing Price vs Time Chart with 20 & 50 Days EMA
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], 'b', label='Closing Price', alpha=0.7)
            plt.plot(df.index, ema20, 'g', label='EMA 20', alpha=0.8)
            plt.plot(df.index, ema50, 'r', label='EMA 50', alpha=0.8)
            plt.title(f"{stock} - Closing Price vs Time (20 & 50 Days EMA)")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ema_chart_path = "static/ema_20_50.png"
            plt.savefig(ema_chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Closing Price vs Time Chart with 100 & 200 Days EMA
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], 'b', label='Closing Price', alpha=0.7)
            plt.plot(df.index, ema100, 'g', label='EMA 100', alpha=0.8)
            plt.plot(df.index, ema200, 'r', label='EMA 200', alpha=0.8)
            plt.title(f"{stock} - Closing Price vs Time (100 & 200 Days EMA)")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ema_chart_path_100_200 = "static/ema_100_200.png"
            plt.savefig(ema_chart_path_100_200, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Plot 3: OHLC Chart
            plt.figure(figsize=(12, 6))
            recent_data = df.tail(50)  # Last 50 days
            plt.plot(recent_data.index, recent_data['Open'], 'g', label='Open', alpha=0.7)
            plt.plot(recent_data.index, recent_data['High'], 'r', label='High', alpha=0.7)
            plt.plot(recent_data.index, recent_data['Low'], 'b', label='Low', alpha=0.7)
            plt.plot(recent_data.index, recent_data['Close'], 'purple', label='Close', alpha=0.8, linewidth=2)
            plt.title(f"{stock} - OHLC Data (Last 50 Days)")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            ohlc_chart_path = "static/ohlc_chart.png"
            plt.savefig(ohlc_chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            # Save dataset as CSV
            csv_file_path = f"static/{stock.replace('.', '_')}_dataset.csv"
            df.to_csv(csv_file_path)

            print("Charts generated successfully and model trained!")

            # Get recent price ranges for the form
            recent_prices = df.tail(10)
            price_ranges = {
                'open_min': float(recent_prices['Open'].min()),
                'open_max': float(recent_prices['Open'].max()),
                'high_min': float(recent_prices['High'].min()),
                'high_max': float(recent_prices['High'].max()),
                'low_min': float(recent_prices['Low'].min()),
                'low_max': float(recent_prices['Low'].max()),
                'volume_min': int(recent_prices['Volume'].min()),
                'volume_max': int(recent_prices['Volume'].max()),
                'latest_close': float(df['Close'].iloc[-1])
            }

            return render_template('index.html', 
                                   stock_symbol=stock,
                                   plot_path_ema_20_50=ema_chart_path, 
                                   plot_path_ema_100_200=ema_chart_path_100_200, 
                                   plot_path_ohlc=ohlc_chart_path,
                                   data_desc=data_desc.to_html(classes='table table-bordered table-striped'),
                                   dataset_link=csv_file_path,
                                   csv_filename=f"{stock.replace('.', '_')}_dataset.csv",
                                   model_trained=True,
                                   price_ranges=price_ranges)
        
        except Exception as e:
            print(f"Error: {str(e)}")
            return render_template('index.html', error=f"An error occurred: {str(e)}")

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predicting closing price"""
    try:
        data = request.get_json()
        
        open_price = float(data.get('open', 0))
        high_price = float(data.get('high', 0))
        low_price = float(data.get('low', 0))
        volume = float(data.get('volume', 0))
        
        # Validate input
        if high_price < max(open_price, low_price) or low_price > min(open_price, high_price):
            return jsonify({'error': 'Invalid price relationships: High should be >= Open/Low, Low should be <= Open/High'}), 400
        
        if any(val <= 0 for val in [open_price, high_price, low_price, volume]):
            return jsonify({'error': 'All values must be positive'}), 400
        
        # Get prediction
        prediction, error = predict_closing_price(open_price, high_price, low_price, volume)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'predicted_close': round(prediction, 2),
            'input_values': {
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'volume': volume
            }
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(f"static/{filename}", as_attachment=True)
    except FileNotFoundError:
        return "File not found", 404

if __name__ == '__main__':
    print("Starting Flask application...")
    print("Make sure you have the following packages installed:")
    print("pip install flask pandas numpy matplotlib yfinance scikit-learn")
   
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)