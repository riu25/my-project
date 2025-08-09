#!/usr/bin/env python3
"""
Example Usage Script for Algo-Trading Prototype
Demonstrates key features and functionality of the trading system.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.automation.trading_engine import TradingEngine
from src.data.data_fetcher import DataFetcher
from src.data.technical_indicators import TechnicalIndicators
from src.strategies.rsi_ma_strategy import RSIMACrossoverStrategy
from src.ml.predictive_model import StockPredictiveModel
from src.utils.sheets_logger import GoogleSheetsLogger
from src.utils.telegram_alerts import TelegramAlertsBot
from src.config import Config


def example_1_data_fetching():
    """Example 1: Data fetching and technical indicators"""
    print("="*60)
    print("EXAMPLE 1: DATA FETCHING & TECHNICAL INDICATORS")
    print("="*60)
    
    # Initialize data fetcher
    data_fetcher = DataFetcher()
    
    # Fetch data for a single stock
    print("Fetching data for RELIANCE.BSE...")
    reliance_data = data_fetcher.get_daily_data("RELIANCE.BSE", outputsize="compact")
    
    if not reliance_data.empty:
        print(f"✅ Fetched {len(reliance_data)} data points")
        print(f"Date range: {reliance_data.index[0].date()} to {reliance_data.index[-1].date()}")
        print(f"Latest close price: ₹{reliance_data['close'].iloc[-1]:.2f}")
        
        # Calculate technical indicators
        print("\nCalculating technical indicators...")
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(reliance_data)
        
        # Display latest indicators
        latest = data_with_indicators.iloc[-1]
        print(f"RSI: {latest['rsi']:.2f}")
        print(f"MACD: {latest['macd']:.4f}")
        print(f"20-day SMA: ₹{latest['sma_20']:.2f}")
        print(f"50-day SMA: ₹{latest['sma_50']:.2f}")
        print(f"Volatility: {latest['volatility']:.4f}")
        
        return data_with_indicators
    else:
        print("❌ Failed to fetch data")
        return None


def example_2_trading_strategy():
    """Example 2: Trading strategy and backtesting"""
    print("\n" + "="*60)
    print("EXAMPLE 2: TRADING STRATEGY & BACKTESTING")
    print("="*60)
    
    # Initialize components
    data_fetcher = DataFetcher()
    strategy = RSIMACrossoverStrategy(initial_capital=100000, risk_per_trade=0.02)
    
    # Fetch historical data
    print("Fetching historical data for backtesting...")
    historical_data = data_fetcher.get_nifty50_data()
    
    if historical_data:
        print(f"✅ Fetched data for {len(historical_data)} stocks")
        
        # Run backtest
        print("Running strategy backtest...")
        results = strategy.backtest(historical_data)
        
        if results:
            print("\n📊 BACKTEST RESULTS:")
            print(f"Period: {results['start_date']} to {results['end_date']}")
            print(f"Initial Capital: ₹{results['initial_capital']:,.2f}")
            print(f"Final Value: ₹{results['final_value']:,.2f}")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Annualized Return: {results['annualized_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Win Rate: {results['win_rate']:.2%}")
            
            # Get current signals
            print("\n📈 CURRENT SIGNALS:")
            current_signals = strategy.get_current_signals(historical_data)
            
            for symbol, signal in current_signals.items():
                if signal['signal'] != 'NONE':
                    print(f"{symbol}: {signal['signal']} (Strength: {signal['strength']})")
                    print(f"  Price: ₹{signal['price']:.2f}, RSI: {signal['rsi']:.2f}")
        
        return results
    else:
        print("❌ Failed to fetch historical data")
        return None


def example_3_ml_prediction():
    """Example 3: Machine learning model training and prediction"""
    print("\n" + "="*60)
    print("EXAMPLE 3: MACHINE LEARNING MODEL")
    print("="*60)
    
    # Initialize components
    data_fetcher = DataFetcher()
    ml_model = StockPredictiveModel(model_type='random_forest')
    
    # Fetch training data
    print("Fetching data for ML model training...")
    training_data = data_fetcher.get_nifty50_data()
    
    if training_data:
        print(f"✅ Fetched training data for {len(training_data)} stocks")
        
        # Train model
        print("Training machine learning model...")
        try:
            training_results = ml_model.train(training_data)
            
            print("\n🤖 ML MODEL RESULTS:")
            print(f"Model Type: {ml_model.model_type}")
            print(f"Training Accuracy: {training_results['train_accuracy']:.4f}")
            print(f"Test Accuracy: {training_results['test_accuracy']:.4f}")
            print(f"Cross-Validation: {training_results['cv_mean']:.4f} ± {training_results['cv_std']:.4f}")
            print(f"Number of Features: {len(ml_model.feature_names)}")
            
            # Feature importance
            if ml_model.feature_importance is not None:
                print("\n🎯 TOP 5 IMPORTANT FEATURES:")
                top_features = ml_model.get_feature_importance(5)
                for _, feature in top_features.iterrows():
                    print(f"  {feature['feature']}: {feature['importance']:.4f}")
            
            # Make prediction on latest data
            print("\n🔮 SAMPLE PREDICTIONS:")
            for symbol, data in list(training_data.items())[:2]:  # First 2 stocks
                prediction = ml_model.predict(data)
                if prediction['prediction'] is not None:
                    direction = "📈 UP" if prediction['prediction'] == 1 else "📉 DOWN"
                    print(f"{symbol}: {direction} (Confidence: {prediction['confidence']:.2%})")
        
        except Exception as e:
            print(f"❌ ML model training failed: {e}")
    
    else:
        print("❌ Failed to fetch training data")


def example_4_integration_testing():
    """Example 4: Test external integrations"""
    print("\n" + "="*60)
    print("EXAMPLE 4: INTEGRATION TESTING")
    print("="*60)
    
    # Test Google Sheets
    print("Testing Google Sheets integration...")
    try:
        sheets_logger = GoogleSheetsLogger()
        if sheets_logger.is_connected():
            print("✅ Google Sheets: Connected")
            
            # Test logging a sample signal
            sample_signal = {
                'symbol': 'TEST.BSE',
                'signal': 'BUY',
                'strength': 'MODERATE',
                'price': 100.50,
                'rsi': 25.5,
                'sma_20': 98.0,
                'sma_50': 95.0
            }
            
            sample_ml_prediction = {
                'prediction': 1,
                'confidence': 0.75
            }
            
            sheets_logger.log_signal(sample_signal, sample_ml_prediction)
            print("✅ Sample signal logged to Google Sheets")
            
        else:
            print("❌ Google Sheets: Not connected")
    except Exception as e:
        print(f"❌ Google Sheets error: {e}")
    
    # Test Telegram Bot
    print("\nTesting Telegram bot integration...")
    try:
        telegram_bot = TelegramAlertsBot()
        if telegram_bot.is_enabled:
            # Send test alert
            test_signal = {
                'symbol': 'TEST.BSE',
                'signal': 'BUY',
                'strength': 'STRONG',
                'price': 100.50,
                'rsi': 25.5,
                'sma_20': 98.0,
                'sma_50': 95.0
            }
            
            success = telegram_bot.send_signal_alert(test_signal)
            if success:
                print("✅ Telegram Bot: Test alert sent")
            else:
                print("❌ Telegram Bot: Failed to send test alert")
        else:
            print("❌ Telegram Bot: Not enabled")
    except Exception as e:
        print(f"❌ Telegram Bot error: {e}")


def example_5_full_engine_demo():
    """Example 5: Full trading engine demonstration"""
    print("\n" + "="*60)
    print("EXAMPLE 5: TRADING ENGINE DEMONSTRATION")
    print("="*60)
    
    # Initialize trading engine
    print("Initializing trading engine...")
    engine = TradingEngine()
    
    # Validate configuration
    print("Validating configuration...")
    validation = engine.validate_configuration()
    print(f"Configuration Status: {validation['overall_status']}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    
    # Get system status
    print("\n📊 SYSTEM STATUS:")
    status = engine.get_system_status()
    
    print(f"Engine Running: {status['engine_status']['running']}")
    print(f"Components Status:")
    for component, is_ok in status['components'].items():
        status_icon = "✅" if is_ok else "❌"
        print(f"  {component}: {status_icon}")
    
    print(f"\nSystem Resources:")
    resources = status['system_resources']
    print(f"  CPU Usage: {resources['cpu_usage']:.1f}%")
    print(f"  Memory Usage: {resources['memory_usage']:.1f}%")
    print(f"  Disk Usage: {resources['disk_usage']:.1f}%")
    
    # Run single market scan
    print("\n🔍 RUNNING MARKET SCAN:")
    scan_results = engine.run_single_scan()
    
    print(f"Timestamp: {scan_results['timestamp']}")
    print(f"Symbols Scanned: {scan_results['symbols_scanned']}")
    print(f"Signals Generated: {scan_results['signals_generated']}")
    print(f"Errors: {scan_results['errors']}")
    
    if scan_results['signals']:
        print("\n📈 DETECTED SIGNALS:")
        for symbol, signal in scan_results['signals'].items():
            if signal['signal'] != 'NONE':
                print(f"  {symbol}: {signal['signal']} (Strength: {signal['strength']})")
                print(f"    Price: ₹{signal['price']:.2f}, RSI: {signal['rsi']:.2f}")
    
    # Generate performance report
    print("\n📋 GENERATING PERFORMANCE REPORT:")
    report = engine.get_performance_report()
    
    if report:
        print("✅ Performance report generated successfully")
        if 'backtest_results' in report and report['backtest_results']:
            bt_results = report['backtest_results']
            print(f"  Strategy Return: {bt_results.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {bt_results.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {bt_results.get('max_drawdown', 0):.2%}")


def main():
    """Run all examples"""
    print("🚀 ALGO-TRADING PROTOTYPE - EXAMPLE USAGE")
    print("=" * 60)
    print("This script demonstrates the key features of the trading system.")
    print("Make sure you have configured your API keys in the .env file.")
    print("=" * 60)
    
    try:
        # Example 1: Data fetching
        data = example_1_data_fetching()
        
        # Example 2: Trading strategy
        backtest_results = example_2_trading_strategy()
        
        # Example 3: ML model
        example_3_ml_prediction()
        
        # Example 4: Integration testing
        example_4_integration_testing()
        
        # Example 5: Full engine demo
        example_5_full_engine_demo()
        
        print("\n" + "="*60)
        print("🎉 ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("1. Configure your API keys in .env file")
        print("2. Run: python main.py config  # Check configuration")
        print("3. Run: python main.py test    # Test integrations")
        print("4. Run: python main.py backtest # Full backtest")
        print("5. Run: python main.py run     # Start trading engine")
        print("\nFor more information, see README.md")
        
    except KeyboardInterrupt:
        print("\n\nExample execution interrupted by user.")
    except Exception as e:
        print(f"\n\nExample execution failed: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()