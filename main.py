"""
AI Crypto Investment Intelligence Platform - Main Entry Point
Provides CLI interface to launch dashboard, API server, or run analysis.
"""

import sys
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

# Fix Unicode encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except (AttributeError, Exception):
        pass

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import API_HOST, API_PORT, MAX_WORKERS, DEFAULT_CRYPTOS
from backend.services.data_collector import CryptoDataCollector
from backend.services.portfolio_manager import PortfolioManager
from utils.helpers import logger, format_currency, format_percentage

def run_api():
    """Launch the FastAPI server."""
    print(f"\n🔌 Launching FastAPI Server on port {API_PORT}...")
    import os
    import uvicorn
    is_dev = os.getenv("ENVIRONMENT", "development").lower() == "development"
    uvicorn.run("backend.api.server:socket_app", host=API_HOST, port=API_PORT, reload=is_dev)


def run_dashboard():
    """Launch the React dashboard."""
    print("\n🖥️ Launching AI Crypto Dashboard...")
    web_dir = Path(__file__).resolve().parent / "web"
    
    if not (web_dir / "node_modules").exists():
        print("📦 node_modules not found. Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=web_dir, shell=True)
    
    # Use shell=True for Windows to find npm/vite
    subprocess.run(["npm", "run", "dev"], cwd=web_dir, shell=True)


def run_analysis():
    """Run comprehensive analysis with parallel processing."""
    print("\n" + "=" * 60)
    print("🔬 Running Full Crypto Analysis (Parallel Processing)")
    print("=" * 60)

    # Load data from MongoDB
    async def load_data():
        from database.mongo_connection import connect_to_mongo
        await connect_to_mongo()
        collector = CryptoDataCollector()
        coins_to_analyze = DEFAULT_CRYPTOS[:5]
        datasets = {}
        for coin_id in coins_to_analyze:
            df = await collector.get_price_history_from_db(coin_id, days=365)
            if not df.empty:
                datasets[coin_id] = df
            else:
                logger.warning(f"No data for {coin_id} in database. Run 'python main.py setup' first.")
        return datasets

    datasets = asyncio.run(load_data())

    if not datasets:
        print("❌ No data available in database. Run 'python main.py setup' first.")
        return

    from ai_models.risk_analyzer import RiskAnalyzer
    from ai_models.predictor import CryptoPricePredictor
    from ai_models.investment_optimizer import InvestmentOptimizer
    from backend.services.report_generator import ReportGenerator

    risk_analyzer = RiskAnalyzer()
    predictor = CryptoPricePredictor()
    optimizer = InvestmentOptimizer()
    report_gen = ReportGenerator()

    # ---- Parallel Risk Analysis ----
    print("\n📊 Running parallel risk analysis...")
    risk_results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(risk_analyzer.analyze_asset_risk, df["price"], coin_id): coin_id
            for coin_id, df in datasets.items()
        }
        for future in as_completed(futures):
            coin_id = futures[future]
            try:
                risk_results[coin_id] = future.result()
            except Exception as e:
                logger.error(f"Risk analysis failed for {coin_id}: {e}")

    for coin_id, result in risk_results.items():
        if "error" not in result:
            print(f"\n  {'='*50}")
            print(f"  📊 {coin_id.upper()}")
            print(f"  Risk Score: {result['risk_score']:.3f} {result['risk_label']}")
            print(f"  Volatility: {result['volatility']['annualized']*100:.1f}%")
            print(f"  Max Drawdown: {result['drawdown']['max_drawdown']*100:.1f}%")
            print(f"  Sharpe Ratio: {result['performance']['sharpe_ratio']:.3f}")

    # ---- Parallel Predictions ----
    print(f"\n🤖 Running parallel ML predictions...")
    prediction_results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                predictor.predict_future_prices, df, coin_id, "random_forest", 14
            ): coin_id
            for coin_id, df in datasets.items()
        }
        for future in as_completed(futures):
            coin_id = futures[future]
            try:
                prediction_results[coin_id] = future.result()
            except Exception as e:
                logger.error(f"Prediction failed for {coin_id}: {e}")

    for coin_id, result in prediction_results.items():
        if "error" not in result:
            print(f"\n  🔮 {coin_id.upper()} Prediction (14 days)")
            print(f"     Current: {format_currency(result['current_price'])}")
            print(f"     Predicted: {format_currency(result['predicted_price'])}")
            print(f"     Change: {format_percentage(result.get('predicted_change_pct', 0))}")
            print(f"     Direction: {result['prediction_direction']}")

    # ---- Portfolio Optimization ----
    print(f"\n📈 Running portfolio optimization...")
    price_data = {coin_id: df["price"] for coin_id, df in datasets.items()}
    opt_result = optimizer.optimize_portfolio(price_data, objective="max_sharpe")

    if "error" not in opt_result:
        print(f"\n  🎯 Optimal Portfolio (Max Sharpe):")
        for alloc in opt_result.get("allocations", []):
            print(f"     {alloc['asset']:15s}: {alloc['weight_pct']:.1f}%")
        print(f"     Expected Return: {opt_result['expected_annual_return']*100:.1f}%")
        print(f"     Volatility: {opt_result['annual_volatility']*100:.1f}%")
        print(f"     Sharpe Ratio: {opt_result['sharpe_ratio']:.3f}")

    # ---- Generate Reports ----
    print(f"\n📋 Generating reports...")
    # Using an async context to call the async portfolio manager
    async def get_portfolio_data():
        pm = PortfolioManager()
        # Find the first portfolio for the main user (PARAMESH_USER_ID)
        from backend.api.server import PARAMESH_USER_ID
        portfolios = await pm.list_portfolios(str(PARAMESH_USER_ID))
        if portfolios:
            return portfolios[0]
        return None

    portfolio = asyncio.run(get_portfolio_data())
    if portfolio:
        report = report_gen.generate_portfolio_report(portfolio, risk_results, prediction_results)
        print(f"   Portfolio report saved: {report.get('csv_path', 'N/A')}")

    pred_report = report_gen.generate_prediction_report(prediction_results)
    print(f"   Prediction report saved: {pred_report.get('csv_path', 'N/A')}")

    # ---- Alert Check ----
    async def check_alerts_async():
        from backend.services.alert_system import AlertSystem
        alert_sys = AlertSystem()
        # Mock market data for alert check
        market_data = [{"coin_id": cid, "price": df["price"].iloc[-1]} for cid, df in datasets.items()]
        return await alert_sys.check_alerts(market_data)

    triggered_alerts = asyncio.run(check_alerts_async())
    if triggered_alerts:
        print(f"\n🔔 Alerts Triggered:")
        for alert in triggered_alerts:
            print(f"   {alert['message']}")

    print("\n" + "=" * 60)
    print("✅ Analysis complete!")
    print("=" * 60)


def run_setup():
    """Initialize the platform with sample data."""
    print("\n" + "=" * 60)
    print("🏗️ Setting up AI Crypto Platform...")
    print("=" * 60)
    
    async def setup_async():
        from database.mongo_connection import connect_to_mongo
        from backend.api.server import seed_initial_data
        await connect_to_mongo()
        await seed_initial_data()
        
    asyncio.run(setup_async())
    print("\n✅ Setup complete! You can now run the API or analysis.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="AI Crypto Investment Intelligence Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  api        Launch the FastAPI REST API server with WebSockets
  analyze    Run full analysis with parallel processing
  setup      Initialize database and seed initial market data
  dashboard  Launch the modern React web dashboard

Examples:
  python main.py api
  python main.py analyze
  python main.py setup
        """
    )
    parser.add_argument("command", nargs="?", default="api",
        choices=["api", "analyze", "setup", "dashboard"],
        help="Command to execute (default: api)")

    args = parser.parse_args()

    if args.command == "api":
        run_api()
    elif args.command == "analyze":
        run_analysis()
    elif args.command == "setup":
        run_setup()
    elif args.command == "dashboard":
        run_dashboard()


if __name__ == "__main__":
    main()
