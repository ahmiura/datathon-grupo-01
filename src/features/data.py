import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from src.config import TICKER, START_DATE, TIMESTEPS, SCALER_PATH

def download_data(ticker=TICKER, start=START_DATE, end=None):
    """Baixa dados históricos do Yahoo Finance."""
    print(f"Baixando dados para {ticker}...")
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']]

def create_sequences(data, timesteps=TIMESTEPS):
    """Cria sequências para o LSTM (Sliding Window)."""
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i:(i + timesteps)])
        y.append(data[i + timesteps])
    return np.array(X), np.array(y)

def preprocess_data(df, train_split_date=None, save_scaler=False):
    """Normaliza os dados e cria as sequências."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    if train_split_date:
        train = df[df.index < train_split_date]
        test = df[df.index >= train_split_date]
        
        scaled_train = scaler.fit_transform(train)
        scaled_test = scaler.transform(test)
        
        if save_scaler:
            joblib.dump(scaler, SCALER_PATH)
            
        X_train, y_train = create_sequences(scaled_train)
        X_test, y_test = create_sequences(scaled_test)
        return X_train, y_train, X_test, y_test, scaler
    else:
        scaled_data = scaler.fit_transform(df)
        if save_scaler:
            joblib.dump(scaler, SCALER_PATH)
        X, y = create_sequences(scaled_data)
        return X, y, scaler