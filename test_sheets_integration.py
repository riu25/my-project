#!/usr/bin/env python3
"""
Test script to verify Google Sheets integration and data logging functionality.
"""

import sys
from pathlib import Path
import os
from datetime import datetime

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_sheets_connection():
    """Test Google Sheets connection and setup"""
    print("=" * 60)
    print("TESTING GOOGLE SHEETS INTEGRATION")
    print("=" * 60)
    
    # Check if credentials are configured
    from src.config import Config
    
    print("1. Checking configuration...")
    print(f"   Credentials file: {Config.GOOGLE_SHEETS_CREDENTIALS_FILE}")
    print(f"   Spreadsheet ID: {Config.GOOGLE_SHEETS_SPREADSHEET_ID}")
    
    if not Config.GOOGLE_SHEETS_CREDENTIALS_FILE or Config.GOOGLE_SHEETS_CREDENTIALS_FILE == "path/to/credentials.json":
        print("\n❌ Google Sheets credentials not configured!")
        print("\nTo fix this:")
        print("1. Create a Google Service Account:")
        print("   - Go to https://console.cloud.google.com/")
        print("   - Create/select a project")
        print("   - Enable Google Sheets API")
        print("   - Create Service Account credentials")
        print("   - Download the JSON file")
        print("\n2. Create a Google Sheet:")
        print("   - Create a new Google Sheet")
        print("   - Share it with your service account email")
        print("   - Copy the spreadsheet ID from the URL")
        print("\n3. Update your .env file:")
        print("   GOOGLE_SHEETS_CREDENTIALS_FILE=/path/to/your/credentials.json")
        print("   GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id")
        return False
    
    if not os.path.exists(Config.GOOGLE_SHEETS_CREDENTIALS_FILE):
        print(f"\n❌ Credentials file not found: {Config.GOOGLE_SHEETS_CREDENTIALS_FILE}")
        return False
    
    # Test connection
    print("\n2. Testing Google Sheets connection...")
    try:
        from src.utils.sheets_logger import GoogleSheetsLogger
        
        sheets_logger = GoogleSheetsLogger()
        
        if sheets_logger.is_connected():
            print("✅ Successfully connected to Google Sheets!")
            
            # Test logging a sample trade
            print("\n3. Testing trade logging...")
            sample_trade = {
                'timestamp': datetime.now(),
                'symbol': 'TEST',
                'action': 'BUY',
                'quantity': 100,
                'entry_price': 150.50,
                'stop_loss': 145.00,
                'take_profit': 160.00,
                'rsi': 25.5,
                'sma_20': 148.75,
                'sma_50': 152.30,
                'signal_strength': 'STRONG'
            }
            
            try:
                sheets_logger.log_trade(sample_trade)
                print("✅ Sample trade logged successfully!")
                
                # Test portfolio summary
                print("\n4. Testing portfolio summary...")
                sample_summary = {
                    'date': datetime.now(),
                    'total_capital': 105000.00,
                    'cash': 85000.00,
                    'positions_value': 20000.00,
                    'total_pnl': 5000.00,
                    'cumulative_return': 0.05,
                    'active_positions': 3,
                    'max_drawdown': -0.02,
                    'win_rate': 0.65
                }
                
                sheets_logger.update_portfolio_summary(sample_summary)
                print("✅ Portfolio summary updated successfully!")
                
                # Test performance metrics
                print("\n5. Testing performance metrics...")
                sample_metrics = {
                    'total_return': 0.05,
                    'annualized_return': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.02,
                    'win_rate': 0.65,
                    'total_trades': 25
                }
                
                sheets_logger.update_performance_metrics(sample_metrics)
                print("✅ Performance metrics updated successfully!")
                
                print("\n" + "=" * 60)
                print("🎉 ALL TESTS PASSED!")
                print("Google Sheets integration is working correctly.")
                print("Your backtest data should now appear in Google Sheets.")
                print("=" * 60)
                return True
                
            except Exception as e:
                print(f"❌ Failed to log data: {e}")
                return False
                
        else:
            print("❌ Failed to connect to Google Sheets")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Google Sheets: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_status():
    """Test configuration status"""
    print("\n" + "=" * 60)
    print("CONFIGURATION STATUS")
    print("=" * 60)
    
    try:
        from src.config import Config
        config_status = Config.validate_config()
        
        print("Configuration validation results:")
        for component, status in config_status.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component}: {'OK' if status else 'MISSING'}")
        
        return all(config_status.values())
        
    except Exception as e:
        print(f"❌ Error checking configuration: {e}")
        return False

def main():
    """Run all tests"""
    try:
        # Test configuration
        config_ok = test_config_status()
        
        if not config_ok:
            print("\n⚠️  Some configuration issues found.")
            print("Please fix the configuration issues above before running backtests.")
            return
        
        # Test Google Sheets
        sheets_ok = test_sheets_connection()
        
        if sheets_ok:
            print("\n🚀 Next steps:")
            print("1. Run: python main.py backtest")
            print("2. Check your Google Sheet for trade logs and performance data")
            print("3. The following tabs should contain data:")
            print("   - Trade_Log: Individual buy/sell transactions")
            print("   - Portfolio_Summary: Portfolio value over time")
            print("   - Performance_Metrics: Overall strategy performance")
            print("   - Signal_Log: Trading signals generated")
        else:
            print("\n❌ Google Sheets integration issues found.")
            print("Please fix the issues above before running backtests.")
    
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()