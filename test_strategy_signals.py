#!/usr/bin/env python3
"""
Test script to debug signal generation in the trading strategy.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_signal_generation():
    """Test signal generation with sample data"""
    print("Testing signal generation logic...")
    
    # Create sample data similar to what the strategy expects
    dates = pd.date_range('2023-01-01', periods=200, freq='D')
    np.random.seed(42)  # For reproducible results
    
    # Generate realistic stock data
    base_price = 100
    prices = []
    current_price = base_price
    
    for i in range(200):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # 0.1% daily return, 2% volatility
        current_price *= (1 + change)
        prices.append(current_price)
    
    # Create OHLCV data
    sample_data = pd.DataFrame({
        'open': [p * np.random.uniform(0.99, 1.01) for p in prices],
        'high': [p * np.random.uniform(1.00, 1.03) for p in prices],
        'low': [p * np.random.uniform(0.97, 1.00) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 200)
    }, index=dates)
    
    print(f"Created sample data with {len(sample_data)} days")
    print(f"Price range: {sample_data['close'].min():.2f} to {sample_data['close'].max():.2f}")
    
    # Calculate technical indicators manually
    def calculate_sma(prices, window):
        return prices.rolling(window=window).mean()
    
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return np.clip(rsi, 0, 100)
    
    # Add technical indicators
    sample_data['sma_20'] = calculate_sma(sample_data['close'], 20)
    sample_data['sma_50'] = calculate_sma(sample_data['close'], 50)
    sample_data['rsi'] = calculate_rsi(sample_data['close'], 14)
    
    # Remove NaN values
    sample_data = sample_data.dropna()
    
    print(f"After adding indicators: {len(sample_data)} valid days")
    print(f"RSI range: {sample_data['rsi'].min():.2f} to {sample_data['rsi'].max():.2f}")
    print(f"SMA 20 vs 50 crossovers: {((sample_data['sma_20'] > sample_data['sma_50']) != (sample_data['sma_20'].shift(1) > sample_data['sma_50'].shift(1))).sum()}")
    
    # Test signal generation logic
    signals_df = sample_data.copy()
    signals_df['signal'] = 0
    
    # Improved buy conditions
    ma_bullish = signals_df['sma_20'] > signals_df['sma_50']
    ma_crossover = (signals_df['sma_20'] > signals_df['sma_50']) & (signals_df['sma_20'].shift(1) <= signals_df['sma_50'].shift(1))
    rsi_oversold = signals_df['rsi'] < 30
    rsi_reasonable = signals_df['rsi'] < 50
    
    buy_condition = (
        (rsi_oversold & ma_bullish) |  # RSI oversold in bullish trend
        (ma_crossover & rsi_reasonable)  # MA crossover with reasonable RSI
    )
    
    # Sell conditions
    ma_bearish = signals_df['sma_20'] < signals_df['sma_50']
    ma_crossover_down = (signals_df['sma_20'] < signals_df['sma_50']) & (signals_df['sma_20'].shift(1) >= signals_df['sma_50'].shift(1))
    rsi_overbought = signals_df['rsi'] > 70
    rsi_high = signals_df['rsi'] > 65
    
    sell_condition = (
        (rsi_overbought) |  # RSI overbought
        (ma_crossover_down) |  # MA crossover down
        (rsi_high & ma_bearish)  # High RSI in bearish trend
    )
    
    # Apply signals
    signals_df.loc[buy_condition, 'signal'] = 1
    signals_df.loc[sell_condition, 'signal'] = -1
    
    # Count signals
    total_buy_signals = buy_condition.sum()
    total_sell_signals = sell_condition.sum()
    
    print(f"\nSignal Results:")
    print(f"Buy signals: {total_buy_signals}")
    print(f"Sell signals: {total_sell_signals}")
    print(f"Signal percentage: {(total_buy_signals + total_sell_signals) / len(signals_df) * 100:.1f}%")
    
    # Show some examples
    if total_buy_signals > 0:
        buy_examples = signals_df[signals_df['signal'] == 1][['close', 'rsi', 'sma_20', 'sma_50']].head(3)
        print(f"\nBuy signal examples:")
        print(buy_examples)
    
    if total_sell_signals > 0:
        sell_examples = signals_df[signals_df['signal'] == -1][['close', 'rsi', 'sma_20', 'sma_50']].head(3)
        print(f"\nSell signal examples:")
        print(sell_examples)
    
    # Check individual conditions
    print(f"\nCondition breakdown:")
    print(f"RSI < 30: {rsi_oversold.sum()}")
    print(f"RSI < 50: {rsi_reasonable.sum()}")
    print(f"RSI > 65: {rsi_high.sum()}")
    print(f"RSI > 70: {rsi_overbought.sum()}")
    print(f"MA bullish (20>50): {ma_bullish.sum()}")
    print(f"MA bearish (20<50): {ma_bearish.sum()}")
    print(f"MA crossover up: {ma_crossover.sum()}")
    print(f"MA crossover down: {ma_crossover_down.sum()}")
    
    return signals_df

def main():
    """Run signal generation test"""
    print("=" * 60)
    print("TESTING STRATEGY SIGNAL GENERATION")
    print("=" * 60)
    
    try:
        signals_df = test_signal_generation()
        
        if signals_df['signal'].abs().sum() > 0:
            print("\n✅ SUCCESS: Signals are being generated!")
            print("The strategy should now produce trades in backtesting.")
        else:
            print("\n❌ ISSUE: No signals generated with sample data.")
            print("The strategy conditions might still be too restrictive.")
        
        print("\n" + "=" * 60)
        print("Next steps:")
        print("1. Run: python main.py backtest")
        print("2. Check if trades are now being generated")
        print("3. Adjust strategy parameters if needed")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()