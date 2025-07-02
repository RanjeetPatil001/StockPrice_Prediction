# StockPrice_Prediction

📈 Stock Price Prediction using LSTM (PyTorch)
This project builds an LSTM-based deep learning model to predict future stock closing prices using historical data. The model is implemented in PyTorch and uses Yahoo Finance for fetching stock data.

🔧 Features
Fetches historical stock prices using yfinance

Scales and processes data using MinMaxScaler

Builds sequences of data for time-series forecasting

Trains a custom LSTM model to predict future closing prices

Visualizes predicted vs actual prices

🧠 Tech Stack
Python

PyTorch

yFinance

Matplotlib

NumPy

Scikit-learn

📦 Installation

pip install torch yfinance scikit-learn matplotlib pandas numpy

▶️ How to Run
Clone the repository or copy the code.

Run the script:

python your_script_name.py

Input the required stock ticker and date range when prompted (e.g., TSLA, 2022-01-01, 2023-01-01).

The model will train and show a plot comparing predicted and actual prices.

📊 Example Output
Final predicted price is printed.

Matplotlib chart shows comparison of actual vs predicted closing prices.

📁 Project Structure
get_stock_data() – Downloads historical data.

prepare_stock_data() – Scales close prices.

LSTMModel – Defines the PyTorch LSTM model.

train_model() – Trains the model on historical data.

make_predictions() – Predicts using the trained model.

📌 Note
Sequence length is fixed to 30 days.

Model is trained for 100 epochs.

Supports GPU (CUDA) if available.
