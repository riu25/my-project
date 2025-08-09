#!/usr/bin/env python3
"""
Main entry point for the Algo-Trading Prototype.
Provides command-line interface for running different components of the trading system.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.automation.trading_engine import TradingEngine
from src.data.data_fetcher import DataFetcher
from src.strategies.rsi_ma_strategy import RSIMACrossoverStrategy
from src.ml.predictive_model import StockPredictiveModel
from src.utils.sheets_logger import GoogleSheetsLogger
from src.utils.telegram_alerts import TelegramAlertsBot
from src.config import Config
from loguru import logger


def setup_logging():
    """Setup logging for the main script."""
    logger.add(
        "logs/main.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB"
    )


def run_engine():
    """Run the main trading engine."""
    logger.info("Starting Algo-Trading Engine...")
    engine = TradingEngine()
    engine.start_automated_trading()


def run_backtest():
    """Run strategy backtesting."""
    logger.info("Running backtest...")
    
    # Initialize components
    data_fetcher = DataFetcher()
    strategy = RSIMACrossoverStrategy()
    
    # Fetch data
    logger.info("Fetching historical data...")
    data = data_fetcher.get_nifty50_data()
    
    if not data:
        logger.error("No data available for backtesting")
        return
    
    # Run backtest
    results = strategy.backtest(data)
    
    # Display results
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)
    print(f"Period: {results.get('start_date')} to {results.get('end_date')}")
    print(f"Total Return: {results.get('total_return', 0):.2%}")
    print(f"Annualized Return: {results.get('annualized_return', 0):.2%}")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2%}")
    print(f"Win Rate: {results.get('win_rate', 0):.2%}")
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Winning Trades: {results.get('winning_trades', 0)}")
    print(f"Losing Trades: {results.get('losing_trades', 0)}")
    print("="*50)
    
    # Plot results
    try:
        strategy.plot_backtest_results(results, save_path="backtest_results.png")
        print("Backtest chart saved as 'backtest_results.png'")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")


def train_ml_model():
    """Train the ML prediction model."""
    logger.info("Training ML model...")
    
    # Initialize components
    data_fetcher = DataFetcher()
    ml_model = StockPredictiveModel(model_type='random_forest')
    
    # Fetch training data
    logger.info("Fetching training data...")
    data = data_fetcher.get_nifty50_data()
    
    if not data:
        logger.error("No data available for training")
        return
    
    # Train model
    results = ml_model.train(data)
    
    # Display results
    print("\n" + "="*50)
    print("ML MODEL TRAINING RESULTS")
    print("="*50)
    print(f"Model Type: {ml_model.model_type}")
    print(f"Training Accuracy: {results.get('train_accuracy', 0):.4f}")
    print(f"Test Accuracy: {results.get('test_accuracy', 0):.4f}")
    print(f"Cross-Validation: {results.get('cv_mean', 0):.4f} ± {results.get('cv_std', 0):.4f}")
    print(f"Number of Features: {len(ml_model.feature_names)}")
    print("="*50)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    ml_model.save_model('models/trading_model.pkl')
    print("Model saved to 'models/trading_model.pkl'")
    
    # Plot analysis
    try:
        ml_model.plot_model_analysis(save_path="ml_model_analysis.png")
        print("Model analysis chart saved as 'ml_model_analysis.png'")
    except Exception as e:
        logger.warning(f"Could not generate plot: {e}")


def scan_market():
    """Run a single market scan."""
    logger.info("Running market scan...")
    
    engine = TradingEngine()
    results = engine.run_single_scan()
    
    print("\n" + "="*50)
    print("MARKET SCAN RESULTS")
    print("="*50)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Symbols Scanned: {results['symbols_scanned']}")
    print(f"Signals Generated: {results['signals_generated']}")
    print(f"Errors: {results['errors']}")
    
    if results['signals']:
        print("\nSIGNALS DETECTED:")
        for symbol, signal in results['signals'].items():
            if signal['signal'] != 'NONE':
                print(f"  {symbol}: {signal['signal']} (Strength: {signal['strength']})")
                print(f"    Price: ₹{signal['price']:.2f}, RSI: {signal['rsi']:.2f}")
    
    print("="*50)


def test_integrations():
    """Test external integrations."""
    logger.info("Testing integrations...")
    
    print("\n" + "="*50)
    print("INTEGRATION TESTS")
    print("="*50)
    
    # Test Alpha Vantage API
    print("Testing Alpha Vantage API...")
    try:
        data_fetcher = DataFetcher()
        test_data = data_fetcher.get_daily_data("RELIANCE.BSE", outputsize="compact")
        if not test_data.empty:
            print("✅ Alpha Vantage API: Connected")
            print(f"   Sample data points: {len(test_data)}")
        else:
            print("❌ Alpha Vantage API: No data received")
    except Exception as e:
        print(f"❌ Alpha Vantage API: {str(e)}")
    
    # Test Google Sheets
    print("\nTesting Google Sheets...")
    try:
        sheets_logger = GoogleSheetsLogger()
        if sheets_logger.is_connected():
            print("✅ Google Sheets: Connected")
            status = sheets_logger.get_connection_status()
            print(f"   Worksheets: {status['worksheets_configured']}")
        else:
            print("❌ Google Sheets: Not connected")
    except Exception as e:
        print(f"❌ Google Sheets: {str(e)}")
    
    # Test Telegram Bot
    print("\nTesting Telegram Bot...")
    try:
        telegram_bot = TelegramAlertsBot()
        if telegram_bot.is_enabled:
            test_result = telegram_bot.test_connection()
            if test_result:
                print("✅ Telegram Bot: Connected")
            else:
                print("❌ Telegram Bot: Connection failed")
        else:
            print("❌ Telegram Bot: Not configured")
    except Exception as e:
        print(f"❌ Telegram Bot: {str(e)}")
    
    print("="*50)


def show_config():
    """Display current configuration."""
    config = Config.get_all_config()
    validation = Config.validate_config()
    
    print("\n" + "="*50)
    print("CONFIGURATION STATUS")
    print("="*50)
    print(f"Status: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
    
    if validation['missing_configs']:
        print(f"Missing: {', '.join(validation['missing_configs'])}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    print(f"\nCapital: ₹{config['initial_capital']:,.2f}")
    print(f"Risk per Trade: {config['risk_per_trade']:.2%}")
    print(f"Max Positions: {config['max_positions']}")
    print(f"Stocks Monitored: {len(config['nifty_stocks'])}")
    
    print("\nStocks:")
    for stock in config['nifty_stocks']:
        print(f"  - {stock}")
    
    print("\nTechnical Parameters:")
    print(f"  RSI Period: {config['rsi_parameters']['period']}")
    print(f"  RSI Oversold: {config['rsi_parameters']['oversold']}")
    print(f"  RSI Overbought: {config['rsi_parameters']['overbought']}")
    print(f"  Short MA: {config['ma_parameters']['short_ma']}")
    print(f"  Long MA: {config['ma_parameters']['long_ma']}")
    
    print("="*50)


def main():
    """Main function with command-line interface."""
    setup_logging()
    
    parser = argparse.ArgumentParser(
        description="Algo-Trading Prototype - RSI + MA Strategy with ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run                 # Start the trading engine
  python main.py backtest           # Run strategy backtesting
  python main.py train-ml           # Train ML prediction model
  python main.py scan               # Run single market scan
  python main.py test               # Test integrations
  python main.py config             # Show configuration
        """
    )
    
    parser.add_argument(
        'command',
        choices=['run', 'backtest', 'train-ml', 'scan', 'test', 'config'],
        help='Command to execute'
    )
    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('cache', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    try:
        if args.command == 'run':
            run_engine()
        elif args.command == 'backtest':
            run_backtest()
        elif args.command == 'train-ml':
            train_ml_model()
        elif args.command == 'scan':
            scan_market()
        elif args.command == 'test':
            test_integrations()
        elif args.command == 'config':
            show_config()
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        print("\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()