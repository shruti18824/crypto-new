"""
Data Collector Module - Fetches real-time and historical cryptocurrency data
from the CoinGecko API and stores it in MongoDB.
"""

import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
from database.mongo_connection import get_database, get_storage_name
from utils.helpers import logger
from config.settings import COINGECKO_BASE_URL, COINGECKO_API_KEY
from ai_models.sentiment_analyzer import SentimentAnalyzer

class CryptoDataCollector:
    """
    Fetches cryptocurrency market data from CoinGecko API.
    Handles rate limiting and MongoDB storage.
    """

    def __init__(self, redis_url: str = None, enable_redis: bool = False):
        self.base_url = COINGECKO_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "CryptoIntelligencePlatform/1.0"
        })
        if COINGECKO_API_KEY:
            self.session.headers["x-cg-demo-api-key"] = COINGECKO_API_KEY
        self._rate_limit_delay = 1.5
        
        # Redis setup
        self.enable_redis = enable_redis
        self.redis = None
        if enable_redis and redis_url:
            try:
                import redis
                self.redis = redis.from_url(redis_url, decode_responses=True)
                logger.info("Redis cache enabled in DataCollector")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.enable_redis = False
        
        self.sentiment_analyzer = SentimentAnalyzer()

    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        url = f"{self.base_url}/{endpoint}"
        try:
            time.sleep(self._rate_limit_delay)
            response = self.session.get(url, params=params, timeout=30)
            
            # Handle rate limiting (429) gracefully with a single retry
            if response.status_code == 429:
                wait_time = 30 # Default wait
                if "Retry-After" in response.headers:
                    try: wait_time = int(response.headers["Retry-After"])
                    except: pass
                logger.warning(f"⚠️ CoinGecko Rate Limit (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                response = self.session.get(url, params=params, timeout=30)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API Request error: {e}")
            return None

    async def fetch_and_store_market_data(self, vs_currency: str = "usd", per_page: int = 50):
        """Fetch market data and store in MongoDB market_data collection."""
        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": 1,
            "sparkline": "true",
            "price_change_percentage": "1h,24h,7d"
        }
        data = await asyncio.to_thread(self._make_request, "coins/markets", params)
        if not data:
            return None
        
        db = get_database()
        timestamp = datetime.utcnow()
        
        market_entries = []
        for coin in data:
            # Fetch sentiment for key coins (to save time/resources)
            sentiment = {"score": 0.0, "label": "Neutral"}
            if per_page <= 10 or coin["id"] in ["bitcoin", "ethereum", "solana"]:
                sentiment = await self.fetch_news_sentiment(coin["id"])
                
            entry = {
                "coin_id": coin["id"],
                "price": coin["current_price"],
                "market_cap": coin["market_cap"],
                "total_volume": coin["total_volume"],
                "change_1h": coin.get("price_change_percentage_1h_in_currency"),
                "change_24h": coin.get("price_change_percentage_24h_in_currency"),
                "change_7d": coin.get("price_change_percentage_7d_in_currency"),
                "sentiment_score": sentiment.get("score", 0.0),
                "sentiment_label": sentiment.get("label", "Neutral"),
                "circulating_supply": coin.get("circulating_supply"),
                "total_supply": coin.get("total_supply"),
                "max_supply": coin.get("max_supply"),
                "sparkline_7d": coin.get("sparkline_in_7d", {}).get("price", []),
                "market_cap_rank": coin.get("market_cap_rank"),
                "timestamp": timestamp,
                "date": timestamp.strftime("%Y-%m-%d")
            }
            market_entries.append(entry)
            
            # Also update cryptocurrency metadata
            await db["cryptocurrencies"].update_one(
                {"coin_id": coin["id"]},
                {"$set": {
                    "symbol": coin["symbol"],
                    "name": coin["name"],
                    "image": coin["image"],
                    "last_updated": timestamp
                }},
                upsert=True
            )
            
        if market_entries:
            # Delete old market data if needed, but we store history here.
            # For simplicity, we just insert.
            await db["market_data"].insert_many(market_entries)
            logger.info(f"Stored {len(market_entries)} market data points in {get_storage_name()}")
            
        return data

    async def get_latest_market_data(self, limit: int = 50) -> List[Dict]:
        """Retrieve the latest market data from Redis (cache) or MongoDB."""
        cache_key = f"market_data_latest_{limit}"
        
        if self.enable_redis and self.redis:
            try:
                import json
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache read error: {e}")

        db = get_database()
        # Find latest timestamp
        latest = await db["market_data"].find_one(sort=[("timestamp", -1)])
        if not latest:
            return []
        
        cursor = db["market_data"].find({"timestamp": latest["timestamp"]}).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        # Batch fetch metadata to eliminate N+1 queries
        coin_ids = [doc["coin_id"] for doc in docs]
        meta_cursor = db["cryptocurrencies"].find({"coin_id": {"$in": coin_ids}})
        meta_dict = {}
        async for m in meta_cursor:
            m.pop("_id", None)
            meta_dict[m["coin_id"]] = m
            
        results = []
        for doc in docs:
            meta = meta_dict.get(doc["coin_id"], {})
            doc.update(meta)
            doc["id"] = str(doc.pop("_id", ""))
            results.append(doc)
            
        # Store in cache
        if self.enable_redis and self.redis and results:
            try:
                import json
                self.redis.setex(cache_key, 60, json.dumps(results)) # Cache for 1 minute
            except Exception as e:
                logger.warning(f"Redis cache write error: {e}")
                
        return results

    async def fetch_historical_prices(self, coin_id: str, days: int = 365):
        """Fetch historical price data from API."""
        endpoint = f"coins/{coin_id}/market_chart"
        params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
        data = self._make_request(endpoint, params)
        if not data or "prices" not in data:
            return None
        
        import pandas as pd
        df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    async def save_price_history_to_db(self, coin_id: str, df):
        """Save historical data to MongoDB efficiently."""
        db = get_database()
        records = df.to_dict("records")
        
        # Prepare market history entries
        history_entries = []
        for rec in records:
            dt = rec["date"]
            history_entries.append({
                "coin_id": coin_id,
                "timestamp": dt,
                "date": dt.strftime("%Y-%m-%d") if hasattr(dt, 'strftime') else str(dt),
                "price": rec["price"]
            })
            
        # We use insert_many to speed up the process instead of update_one in a loop
        if history_entries:
            await db["market_data"].insert_many(history_entries)
            logger.info(f"Saved {len(history_entries)} history points for {coin_id}")

    async def get_price_history_from_db(self, coin_id: str, days: int = 365):
        """Get price history from MongoDB."""
        db = get_database()
        import pandas as pd
        start_date = datetime.utcnow() - timedelta(days=days)
        # Explicitly exclude _id to avoid serialization issues
        cursor = db["market_data"].find(
            {
                "coin_id": coin_id,
                "timestamp": {"$gte": start_date}
            },
            {"_id": 0}
        ).sort("timestamp", 1)
        
        results = []
        async for doc in cursor:
            # Manually remove any lingering ObjectId just in case
            if "_id" in doc:
                del doc["_id"]
            results.append(doc)
        print(f"DEBUG: get_price_history_from_db for {coin_id} returned {len(results)} docs")
        return pd.DataFrame(results)

    def fetch_trending(self):
        """Fetch trending coins from API."""
        data = self._make_request("search/trending")
        if data and 'coins' in data:
            # The trending endpoint returns a list of items with a nested 'item' key
            coin_ids = [c['item']['id'] for c in data['coins']]
            return self.fetch_market_data_by_ids(coin_ids)
        return []

    def fetch_market_data_by_ids(self, coin_ids: List[str]):
        """Fetch full market data for a specific list of coin IDs."""
        if not coin_ids:
            return []
        params = {
            "vs_currency": "usd",
            "ids": ",".join(coin_ids),
            "order": "market_cap_desc",
            "sparkline": "true",
            "price_change_percentage": "1h,24h,7d"
        }
        return self._make_request("coins/markets", params)

    def fetch_by_category(self, category: str):
        """Fetch coins based on a specific category (e.g., 'solana-ecosystem')."""
        params = {
            "category": category
        }
        data = self._make_request("coins/markets", params)
        return data

    async def fetch_news_sentiment(self, coin_id: str):
        """
        Fetch latest news for a coin and analyze sentiment.
        Uses a public RSS feed or mock aggregator for demonstration.
        """
        # For demonstration, we'll use a mix of real RSS (if possible) and news headlines
        # In a production app, you'd use NewsAPI.org or CryptoPanic API.
        try:
            # Mock news items for demonstration
            mock_news = [
                f"{coin_id.capitalize()} price expected to surge as institutional adoption grows.",
                f"New regulatory clarity brings confidence to the {coin_id} market.",
                f"Trading volume for {coin_id} hits all-time high amidst positive sentiment.",
                f"Concerns over network congestion temporarily dampen {coin_id} outlook.",
                f"Whale activity suggests a major breakout for {coin_id} is coming."
            ]
            
            # Real world logic would be:
            # response = self.session.get(f"https://cryptopanic.com/api/v1/posts/?auth_token=TOKEN&currencies={coin_id}")
            # headlines = [post['title'] for post in response.json()['results']]
            
            sentiment = self.sentiment_analyzer.aggregate_sentiment(mock_news)
            return sentiment
        except Exception as e:
            logger.error(f"Error fetching sentiment for {coin_id}: {e}")
            return {"score": 0.0, "label": "Neutral", "count": 0}

    def fetch_global_data(self):
        """Fetch global market data from CoinGecko API."""
        return self._make_request("global")

    def fetch_fear_and_greed_index(self):
        """Fetch Fear & Greed Index from alternative.me API."""
        try:
            res = requests.get("https://api.alternative.me/fng/", timeout=10)
            if res.ok:
                return res.json()["data"][0]
            return None
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed: {e}")
            return None

    async def get_market_summary(self):
        """Combine global market stats for the dashboard header with caching."""
        cache_key = "market_summary"
        
        if self.enable_redis and self.redis:
            try:
                import json
                cached = self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Redis cache summary read error: {e}")

        global_data = await asyncio.to_thread(self.fetch_global_data)
        fng_data = await asyncio.to_thread(self.fetch_fear_and_greed_index)
        
        # In mock mode, we'll generate some realistic values if APIs fail
        if not global_data:
            global_data = {
                "data": {
                    "total_market_cap": {"usd": 2310000000000},
                    "market_cap_change_percentage_24h_usd": -0.93,
                    "market_cap_percentage": {"btc": 58.4, "eth": 10.3}
                }
            }
        if not fng_data:
            fng_data = {"value": "18", "value_classification": "Extreme Fear"}
            
        # Altcoin season logic: 100 - BTC Dominance (simple proxy)
        btc_dominance = global_data.get("data", {}).get("market_cap_percentage", {}).get("btc", 50)
        altcoin_season = 100 - btc_dominance
        alt_label = "Altcoin" if altcoin_season > 50 else "Bitcoin"

        # Calculate a mock Average RSI based on 24h change
        market_change = global_data.get("data", {}).get("market_cap_change_percentage_24h_usd", 0)
        avg_rsi = 50 + (market_change * 5) # Simple mapping for visual consistency
        avg_rsi = max(0, min(100, avg_rsi))
        rsi_label = "Overbought" if avg_rsi > 70 else ("Oversold" if avg_rsi < 30 else "Neutral")

        res = {
            "market_cap": global_data.get("data", {}).get("total_market_cap", {}).get("usd"),
            "market_cap_change": global_data.get("data", {}).get("market_cap_change_percentage_24h_usd"),
            "dominance": global_data.get("data", {}).get("market_cap_percentage", {}),
            "fear_and_greed": fng_data,
            "altcoin_season": {"value": f"{altcoin_season:.0f}/100", "label": alt_label},
            "avg_rsi": {"value": f"{avg_rsi:.2f}", "label": rsi_label}
        }
        
        if self.enable_redis and self.redis:
            try:
                import json
                self.redis.setex(cache_key, 300, json.dumps(res)) # Cache for 5 minutes
            except Exception as e:
                logger.warning(f"Redis cache summary write error: {e}")
                
        return res

    def fetch_coin_details(self, coin_id: str):
        """Fetch detailed information for a specific cryptocurrency."""
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "false",
            "developer_data": "false"
        }
        return self._make_request(f"coins/{coin_id}", params)

def generate_sample_data(coins, days=365):
    """Generate mock historical data for development purposes."""
    datasets = {}
    timestamp = datetime.utcnow()
    for coin_id in coins:
        data = []
        base_price = 1000.0 if coin_id == "bitcoin" else 100.0
        for i in range(days):
            price = base_price * (1 + (i % 10) / 100.0)
            data.append({
                "coin_id": coin_id,
                "price": price,
                "market_cap": price * 1000000,
                "total_volume": price * 50000,
                "change_24h": 1.5,
                "timestamp": timestamp - timedelta(days=days-i)
            })
        datasets[coin_id] = data
    return datasets
