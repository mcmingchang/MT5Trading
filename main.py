import time, json, os, talib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import MetaTrader5 as mt5


def load_mt5():
    file_name = 'trading_config.json'
    if os.path.exists(file_name):
        with open(file_name, 'r') as f:
            data = json.load(f)
    mt5_path = data.get('mt5_path', "C:/Program Files/MetaTrader 5 EXNESS/terminal64.exe")
    login = data.get('login', 123)  # 账户
    password = data.get('password', '')  # 密码
    server = data.get('server', "Exness-MT5Trial5")  # 服务器
    mt5.initialize(path=mt5_path, login=login, password=password, server=server)


def calculate_ma_values(df, periods=(5, 10, 15, 30, 60)):
    df = df.sort_index()  # 确保数据按时间排序
    ma_values = {}
    current_price = df['close'].iloc[-1]
    for period in periods:
        ma_column = f'ma_{period}'
        df[ma_column] = talib.SMA(df['close'], timeperiod=period)
        ma_values[period] = df[ma_column].iloc[-1] if not pd.isna(df[ma_column].iloc[-1]) else current_price

    signals = {'alignment': 'neutral', 'trend': 'neutral', 'strength': 0}

    for ma_k, ma_v in ma_values.items():
        if current_price > ma_v:
            signals[ma_k] = 'above'  # 价格在均线上方
            signals['strength'] += 1
        else:
            signals[ma_k] = 'below'  # 价格在均线下方
            signals['strength'] -= 1
    # # MA排列判断   待优化
    # if current_ma_s > current_ma_m > current_ma_l:
    #     signals['alignment'] = 'bullish_alignment'
    #     signals['trend'] = 'bullish'
    #     signals['strength'] += 2
    # elif current_ma_s < current_ma_m < current_ma_l:
    #     signals['alignment'] = 'bearish_alignment'
    #     signals['trend'] = 'bearish'
    #     signals['strength'] += 2

    return ma_values, signals

def calculate_rsi_signals(df, rsi_period=14, rsi_overbought=70, rsi_oversold=30):
    """计算当前RSI信号 - 包含超买超卖和背离"""

    rsi = talib.RSI(df['close'], timeperiod=rsi_period)
    df['rsi'] = rsi
    current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

    signals = {
        'value': current_rsi,
        'condition': 'neutral',  # 状况：默认中性
        'momentum': 'neutral',  # 动量：默认中性
        'strength': 0  # 强度
    }  # 计算当前rsi

    if current_rsi > rsi_overbought:
        signals['condition'] = 'overbought'  # 超买
        signals['strength'] += 2
    elif current_rsi < rsi_oversold:
        signals['condition'] = 'oversold'  # 超卖
        signals['strength'] += 2
    elif current_rsi > 50:
        signals['condition'] = 'bullish'  # 看涨
        signals['strength'] += 1
    else:
        signals['condition'] = 'bearish'  # 看跌
        signals['strength'] += 1

    # 动量判断
    if len(rsi) > 1:
        prev_rsi = rsi.iloc[-2] if not pd.isna(rsi.iloc[-2]) else 50
        if current_rsi > prev_rsi:
            signals['momentum'] = 'rising'  # 上升
            signals['strength'] += 1
        else:
            signals['momentum'] = 'falling'  # 下跌
            signals['strength'] += 1
    return signals

def calculate_atr_values(df, period=14):
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()
    df['atr'] = atr
    return atr.iloc[-1]

def calculate_macd_signals(df, macd_fast=12, macd_slow=26, macd_signal=9):
    """计算MACD信号 - 包含金叉死叉和背离检测"""
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=macd_fast,
                                              slowperiod=macd_slow, signalperiod=macd_signal)
    df['macd'], df['macd_signal'], df['macd_hist'] = macd, macd_signal, macd_hist
    # 获取最新值
    current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
    current_signal = macd_signal.iloc[-1] if not pd.isna(macd_signal.iloc[-1]) else 0
    current_hist = macd_hist.iloc[-1] if not pd.isna(macd_hist.iloc[-1]) else 0

    # MACD信号逻辑
    signals = {
        'macd': current_macd,
        'macd_signal': current_signal,
        'macd_hist': current_hist,
        'trend': 'neutral',
        'momentum': 'neutral',
        'crossover': 'none',
        'strength': 0
    }

    # 金叉死叉判断
    if (current_macd > current_signal and
            macd.iloc[-2] <= macd_signal.iloc[-2]):
        signals['crossover'] = 'golden_cross'  # 金叉
        signals['trend'] = 'bullish'  # 看涨
        signals['strength'] += 2
    elif (current_macd < current_signal and
          macd.iloc[-2] >= macd_signal.iloc[-2]):
        signals['crossover'] = 'death_cross'  # 死叉
        signals['trend'] = 'bearish'  # 看跌
        signals['strength'] += 2

    # 零轴位置判断
    if current_macd > 0 and current_signal > 0:
        signals['trend'] = 'bullish'  # 看涨
        signals['strength'] += 1
    elif current_macd < 0 and current_signal < 0:
        signals['trend'] = 'bearish'  # 看跌
        signals['strength'] += 1

    # 动量判断
    if current_hist > 0 and current_hist > macd_hist.iloc[-2] if len(macd_hist) > 1 else current_hist:
        signals['momentum'] = 'increasing'  # 增加
        signals['strength'] += 1
    elif current_hist < 0:
        signals['momentum'] = 'decreasing'  # 兼容
        signals['strength'] += 1
    return signals

def calculate_bollinger_bands(df, period=20, num_std=2.0):
    """计算布林带"""
    sma = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    df['bb_middle'], df['bb_upper'], df['bb_lower'] = sma, upper, lower
    return float(sma.iloc[-1]), float(upper.iloc[-1]), float(lower.iloc[-1])

def analyze_ma_arrangement(ma_values, current_price):
    # 分析移动平均线的排列情况
    sorted_periods = sorted(ma_values.keys())
    ma_prices = [ma_values[period] for period in sorted_periods]
    is_bullish = all(ma_prices[i] > ma_prices[i + 1] for i in range(len(ma_prices) - 1))  # 检查多头排列（短期 > 中期 > 长期）
    is_bearish = all(ma_prices[i] < ma_prices[i + 1] for i in range(len(ma_prices) - 1))  # 检查空头排列（短期 < 中期 < 长期）

    # 检查价格与均线关系
    above_ma_count = sum(1 for period in sorted_periods if current_price > ma_values[period])
    if is_bullish:
        print(f"📈 趋势信号: 强势多头, 价格在 {above_ma_count}/{len(sorted_periods)} 条均线之上")
    elif is_bearish:
        print(f"📉 趋势信号: 强势空头, 价格在 {above_ma_count}/{len(sorted_periods)} 条均线之上")
    else:
        print(f"➡️  趋势信号: 震荡行情, 价格在 {above_ma_count}/{len(sorted_periods)} 条均线之上")

    # 支撑阻力分析
    support_level = min(ma_values.values())
    resistance_level = max(ma_values.values())
    print(f"🛡️  最近支撑: {support_level:.2f} (MA{min(ma_values, key=ma_values.get)})")
    print(f"🎯 最近阻力: {resistance_level:.2f} (MA{max(ma_values, key=ma_values.get)})")


def print_ma_analysis(ma_values, df, k_name):
    """
    格式化打印移动平均线分析结果
    """
    print("=" * 60)
    print(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, {len(df)} 根K线")
    print(f"{k_name}最新价格: {df['Close'].iloc[-1]:.2f}")
    print("-" * 60)
    for period in sorted(ma_values.keys()):
        current_price = df['Close'].iloc[-1]
        ma_value = ma_values[period]
        price_vs_ma = current_price - ma_value
        print(f"MA{period:3d}: {ma_value:8.2f} |  差值: {price_vs_ma:7.2f}")
    print("-" * 60)
    analyze_ma_arrangement(ma_values, df['Close'].iloc[-1])  # 分析均线排列

def build_prompt(xau_info, all_timeframes):
    prompt = f'伦敦金，当前价格:{xau_info["bid"]}'
    for k_name, v_data in all_timeframes.items():
        prompt += f"\n【{k_name}周期】\n"
        ma_value = v_data['ma']
        for k,v in ma_value.items():
            prompt += f"MA{k}:{round(v, 2)} | "
        prompt += '\n'
        prompt += f"BOLL上轨:{round(v_data['bb_upper'], 2)} | BOLL中轨:{round(v_data['bb_middle'], 2)} | BOLL下轨:{round(v_data['bb_lower'], 2)} | ATR:{round(v_data['atr'], 2)}\n"
        prompt += f"RSI:{round(v_data['rsi'], 2)} | MACD:{round(v_data['macd'], 2)} | MACD_SIGNAL:{round(v_data['macd_signal'], 2)} | MACD_HIST:{round(v_data['macd_hist'], 2)}\n"
    return prompt


if __name__ == '__main__':
    load_mt5()
    symbols_data = {'黄金': 'XAUUSDm', }

    timeframes = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                  'M30': mt5.TIMEFRAME_M30, 'H1': mt5.TIMEFRAME_H1,
                  'H4': mt5.TIMEFRAME_H4, 'D1': mt5.TIMEFRAME_D1}

    info = mt5.symbol_info('XAUUSDm')
    xau_info = {"symbol": 'XAUUSDm', "bid": round(info.bid, 2), "ask": round(info.ask, 2),
                "spread": round(info.ask - info.bid, 4),
                "digits": info.digits, "volume_min": info.volume_min, "volume_max": info.volume_max,
                "volume_step": info.volume_step}

    all_timeframes = {}
    for k_name, timeframe in timeframes.items():
        rates = mt5.copy_rates_from_pos("XAUUSDm", timeframe, 0, 250)
        if rates is None or len(rates) == 0:
            print(f"No data received for timeframe {timeframe}")

        df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'tick_volume'])
        if df.empty:
            print("Empty dataframe received")

        df['time'] = pd.to_datetime(df['time'], unit='s', cache=True)  # 转换datetime
        df.set_index('time', inplace=True)
        del rates  # 立即清理不需要的数据
        df = df.rename(columns={'tick_volume': 'volume'})  # 成交量改名
        df.sort_index(inplace=True)

        ma_values, ma_signals = calculate_ma_values(df, periods=(5, 10, 15, 30, 60))  # 计算ma 120，240
        rsi = calculate_rsi_signals(df)  # 计算rsi
        macd = calculate_macd_signals(df)  # 计算macd
        atr = calculate_atr_values(df)  # 计算布林带
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df)
        all_timeframes[k_name] = {'ma': ma_values, 'rsi': rsi['value'],
                                  'macd': macd['macd'], 'macd_signal': macd['macd_signal'],
                                  'macd_hist': macd['macd_hist'],
                                  'atr': atr, 'bb_middle': bb_middle, 'bb_upper': bb_upper, 'bb_lower': bb_lower}

    prompt = build_prompt(xau_info, all_timeframes)
    print(prompt)
