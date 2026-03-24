"""
AI Prediction Engine - Modular Wrapper
Refactored to delegate to specialized modules: 
data_preprocessing, model_training, prediction_engine, and evaluation.
"""

import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path
import yfinance as yf

# Modular imports
from ai_models.pipeline import AIPredictionPipeline
from utils.helpers import logger

class CryptoPricePredictor:
    """
    Wrapper for the modular AI Prediction Pipeline.
    Maintains compatibility with existing server API.
    """
    def __init__(self):
        self.pipeline = AIPredictionPipeline()

    def get_ohlcv_data(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        """Fetch clean OHLCV data using yfinance."""
        symbol_map = {
            "bitcoin": "BTC-USD",
            "ethereum": "ETH-USD",
            "solana": "SOL-USD",
            "cardano": "ADA-USD",
        }
        symbol = symbol_map.get(coin_id, f"{coin_id.upper()}-USD")
        
        try:
            logger.info(f"Fetching OHLCV data for {symbol} from yfinance...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{days}d")
            
            if df.empty:
                return pd.DataFrame()
                
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "price",
                "Volume": "total_volume"
            })
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            return df
        except Exception as e:
            logger.error(f"Error fetching data from yfinance: {e}")
            return pd.DataFrame()

    def _format_prediction(self, prediction: Dict) -> Dict:
        """Standardize prediction format for frontend and CLI."""
        if "forecast" in prediction:
            # Ensure 'predictions' list exists for Recharts
            if "predictions" not in prediction:
                prediction["predictions"] = [{"date": p["date"], "predicted_price": p["price"]} for p in prediction["forecast"]]
            
            # Ensure 'predicted_price' exists (use last forecast point)
            if "predicted_price" not in prediction:
                prediction["predicted_price"] = prediction["forecast"][-1]["price"]
            
            # Add derived metrics
            current = prediction.get("current_price", 0)
            if current > 0:
                change_pct = (prediction["predicted_price"] / current - 1) * 100
                prediction["predicted_change_pct"] = change_pct
                prediction["prediction_direction"] = "Bullish" if change_pct > 2 else ("Bearish" if change_pct < -2 else "Neutral")
            else:
                prediction["predicted_change_pct"] = 0
                prediction["prediction_direction"] = "Neutral"
                
        return prediction

    def predict_future_prices(self, df: pd.DataFrame, coin_id: str, 
                              model_type: str = "random_forest", 
                              days_ahead: int = 7) -> Dict:
        """Compatibility method for single-model prediction."""
        prediction = self.pipeline.get_prediction(df, coin_id, days_ahead)
        if "error" in prediction:
            self.pipeline.run_training_cycle(df, coin_id)
            prediction = self.pipeline.get_prediction(df, coin_id, days_ahead)
            
        return self._format_prediction(prediction)

    def ensemble_predict(self, df: pd.DataFrame, coin_id: str, days_ahead: int = 7) -> Dict:
        """Delegate to the modular pipeline ensemble."""
        prediction = self.pipeline.get_prediction(df, coin_id, days_ahead)
        if "error" in prediction:
            self.pipeline.run_training_cycle(df, coin_id)
            prediction = self.pipeline.get_prediction(df, coin_id, days_ahead)
            
        return self._format_prediction(prediction)
