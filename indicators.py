"""
高度なテクニカル指標とFVG分析モジュール
TA-Lib依存を削除し、pandas-taを使用
"""
import pandas as pd
import numpy as np
# pandas-ta 0.3.14b0 uses np.NaN, which is deprecated.
# This is a workaround for a known issue in pandas-ta.
# See: https://github.com/twopirllc/pandas-ta/issues/561
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
from typing import Dict, List, Tuple, Optional
import pandas_ta as ta

class AdvancedIndicators:
    """高度なテクニカル指標クラス"""

    @staticmethod
    def calculate_market_structure(df: pd.DataFrame) -> pd.DataFrame:
        """市場構造（Higher High, Higher Low等）を分析"""
        df = df.copy()

        # スイングハイ・ローを検出
        df['SwingHigh'] = df['High'].where(
            (df['High'] > df['High'].shift(1)) &
            (df['High'] > df['High'].shift(-1))
        )
        df['SwingLow'] = df['Low'].where(
            (df['Low'] < df['Low'].shift(1)) &
            (df['Low'] < df['Low'].shift(-1))
        )

        # HH, HL, LH, LLを判定
        df['MarketStructure'] = 'Neutral'

        # 前回の高値・安値と比較
        prev_high = df['SwingHigh'].dropna().shift(1).reindex(df.index)
        prev_low = df['SwingLow'].dropna().shift(1).reindex(df.index)

        # トレンド判定
        for i in range(1, len(df)):
            current_swing_high = df['SwingHigh'].iloc[i]
            previous_high_val = prev_high.iloc[i]

            current_swing_low = df['SwingLow'].iloc[i]
            previous_low_val = prev_low.iloc[i]

            if not pd.isna(current_swing_high):
                if not pd.isna(previous_high_val):
                    if current_swing_high > previous_high_val:
                        df.loc[df.index[i], 'MarketStructure'] = 'HH'  # Higher High
                    else:
                        df.loc[df.index[i], 'MarketStructure'] = 'LH'  # Lower High

            if not pd.isna(current_swing_low):
                if not pd.isna(previous_low_val):
                    if current_swing_low > previous_low_val:
                        df.loc[df.index[i], 'MarketStructure'] = 'HL'  # Higher Low
                    else:
                        df.loc[df.index[i], 'MarketStructure'] = 'LL'  # Lower Low

        return df

    @staticmethod
    def detect_imbalance_zones(df: pd.DataFrame, min_imbalance_percent: float = 0.3) -> List[Dict]:
        """インバランスゾーン（FVGを含む）を検出"""
        imbalances = []

        for i in range(2, len(df)):
            # FVG（Fair Value Gap）
            gap_up = df['Low'].iloc[i] - df['High'].iloc[i-2]
            if gap_up > 0 and (gap_up / df['Close'].iloc[i-1]) * 100 >= min_imbalance_percent:
                imbalances.append({
                    'type': 'FVG_Bullish',
                    'date': df.index[i],
                    'top': df['Low'].iloc[i],
                    'bottom': df['High'].iloc[i-2],
                    'size_percent': (gap_up / df['Close'].iloc[i-1]) * 100
                })

            # ベアリッシュFVG
            gap_down = df['Low'].iloc[i-2] - df['High'].iloc[i]
            if gap_down > 0 and (gap_down / df['Close'].iloc[i-1]) * 100 >= min_imbalance_percent:
                imbalances.append({
                    'type': 'FVG_Bearish',
                    'date': df.index[i],
                    'top': df['Low'].iloc[i-2],
                    'bottom': df['High'].iloc[i],
                    'size_percent': (gap_down / df['Close'].iloc[i-1]) * 100
                })

            # Order Block検出（強い方向性を持つローソク足）
            body_size = abs(df['Close'].iloc[i] - df['Open'].iloc[i])
            candle_range = df['High'].iloc[i] - df['Low'].iloc[i]

            if candle_range > 0 and body_size / candle_range > 0.7:  # 実体が70%以上
                if df['Close'].iloc[i] > df['Open'].iloc[i]:  # 陽線
                    imbalances.append({
                        'type': 'OrderBlock_Bullish',
                        'date': df.index[i],
                        'top': df['High'].iloc[i],
                        'bottom': df['Low'].iloc[i],
                        'strength': body_size / candle_range
                    })

        return imbalances

    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 20) -> Dict:
        """ボリュームプロファイルを計算"""
        price_range = df['High'].max() - df['Low'].min()
        bin_size = price_range / bins

        volume_profile = {}

        for i in range(bins):
            price_low = df['Low'].min() + (i * bin_size)
            price_high = price_low + bin_size

            # この価格帯での出来高を集計
            mask = ((df['High'] >= price_low) & (df['Low'] <= price_high))
            volume_in_range = df.loc[mask, 'Volume'].sum()

            volume_profile[f"{price_low:.2f}-{price_high:.2f}"] = {
                'volume': volume_in_range,
                'price_center': (price_low + price_high) / 2
            }

        # POC（Point of Control）を特定
        poc_range = max(volume_profile.items(), key=lambda x: x[1]['volume'])

        return {
            'profile': volume_profile,
            'poc': poc_range[1]['price_center'],
            'poc_volume': poc_range[1]['volume']
        }

    @staticmethod
    def calculate_order_flow(df: pd.DataFrame) -> pd.DataFrame:
        """オーダーフロー指標を計算"""
        df = df.copy()

        # Delta（買い圧力 - 売り圧力）を推定
        df['Delta'] = np.where(
            df['Close'] > df['Open'],
            df['Volume'] * ((df['Close'] - df['Low']) / (df['High'] - df['Low'])),
            -df['Volume'] * ((df['High'] - df['Close']) / (df['High'] - df['Low']))
        )

        # 累積Delta
        df['CumulativeDelta'] = df['Delta'].cumsum()

        # Delta移動平均
        df['DeltaMA'] = df['Delta'].rolling(window=10).mean()

        return df

    @staticmethod
    def detect_liquidity_zones(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
        """流動性ゾーン（ストップ狩りゾーン）を検出"""
        liquidity_zones = []
        
        # 直近の高値・安値を検出
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]

            # スイングハイ
            swing_highs = []
            for j in range(2, len(window)-2):
                if (window['High'].iloc[j] > window['High'].iloc[j-1] and
                    window['High'].iloc[j] > window['High'].iloc[j-2] and
                    window['High'].iloc[j] > window['High'].iloc[j+1] and
                    window['High'].iloc[j] > window['High'].iloc[j+2]):
                    swing_highs.append({
                        'price': window['High'].iloc[j],
                        'date': window.index[j]
                    })

            # 複数回テストされた価格帯を特定
            for high in swing_highs:
                touch_count = len([h for h in swing_highs
                                 if abs(h['price'] - high['price']) / high['price'] < 0.005])
                if touch_count >= 2:
                    liquidity_zones.append({
                        'type': 'Resistance_Liquidity',
                        'price': high['price'],
                        'strength': touch_count,
                        'last_test': high['date']
                    })

        # 重複を除去
        unique_zones = []
        for zone in liquidity_zones:
            if not any(abs(z['price'] - zone['price']) / zone['price'] < 0.005
                      for z in unique_zones):
                unique_zones.append(zone)

        return unique_zones

    @staticmethod
    def calculate_smart_money_index(df: pd.DataFrame) -> pd.Series:
        """スマートマネー指標（機関投資家の動き）を計算"""
        # 最初の30分と最後の30分の動きを比較
        # ※簡易版：日足データでは前日終値と当日終値の関係で代用

        df = df.copy()

        # 出来高加重価格（VWAP）
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])

        # スマートマネーの動き（終値とVWAPの乖離）
        df['SMI'] = ((df['Close'] - df['VWAP']) / df['VWAP']) * 100

        # 大口の動き（出来高急増時の価格変動）
        df['VolumeRatio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        df['SmartMoneyFlow'] = np.where(
            df['VolumeRatio'] > 1.5,
            df['SMI'] * df['VolumeRatio'],
            df['SMI']
        )

        return df['SmartMoneyFlow']

    @staticmethod
    def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """モメンタム系指標を追加（pandas-ta版）"""
        df = df.copy()

        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)

        # MACD
        macd_result = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd_result is not None:
            df['MACD'] = macd_result['MACD_12_26_9']
            df['MACD_Signal'] = macd_result['MACDs_12_26_9']
            df['MACD_Hist'] = macd_result['MACDh_12_26_9']

        # ストキャスティクス
        stoch_result = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
        if stoch_result is not None:
            df['SlowK'] = stoch_result[f'STOCHk_14_3_3']
            df['SlowD'] = stoch_result[f'STOCHd_14_3_3']

        # ATR（ボラティリティ）
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)

        return df

class FVGAnalyzer:
    """FVG専門分析クラス"""

    @staticmethod
    def classify_fvg(fvg: Dict, df: pd.DataFrame) -> str:
        """FVGの質を分類"""
        # FVG発生時の状況を分析
        fvg_date = fvg['date']
        idx = df.index.get_loc(fvg_date)

        # 出来高チェック
        volume_avg = df['Volume'].iloc[idx-20:idx].mean()
        current_volume = df['Volume'].iloc[idx]
        volume_ratio = current_volume / volume_avg

        # トレンド方向
        sma20 = df['Close'].iloc[idx-20:idx].mean()
        sma50 = df['Close'].iloc[idx-50:idx].mean() if idx >= 50 else sma20
        trend_strength = (sma20 - sma50) / sma50

        # FVGサイズ
        gap_size = fvg['size_percent']

        # 分類
        score = 0
        if volume_ratio > 2.0:
            score += 3
        elif volume_ratio > 1.5:
            score += 2
        elif volume_ratio > 1.2:
            score += 1

        if trend_strength > 0.02:
            score += 2
        elif trend_strength > 0:
            score += 1

        if gap_size > 1.0:
            score += 2
        elif gap_size > 0.7:
            score += 1

        if score >= 5:
            return "Premium"  # 最高品質
        elif score >= 3:
            return "Standard"  # 標準
        else:
            return "Weak"  # 弱い

    @staticmethod
    def calculate_fvg_fill_probability(fvg: Dict, current_price: float,
                                     market_data: pd.DataFrame) -> float:
        """FVGが埋められる確率を計算"""
        # 簡易的な確率計算
        gap_top = fvg['top'] # Changed from gap_top to top
        gap_bottom = fvg['bottom'] # Changed from gap_bottom to bottom
        gap_center = (gap_top + gap_bottom) / 2

        # 現在価格からの距離
        distance_ratio = abs(current_price - gap_center) / gap_center

        # 基本確率
        base_prob = 0.7  # 統計的に70%のFVGは埋められる

        # 距離による調整
        if distance_ratio < 0.02:  # 2%以内
            prob = base_prob * 1.2
        elif distance_ratio < 0.05:  # 5%以内
            prob = base_prob
        else:
            prob = base_prob * (1 - distance_ratio)

        # トレンドによる調整
        if len(market_data) > 50:
            trend = (market_data['Close'].iloc[-1] - market_data['Close'].iloc[-50]) / market_data['Close'].iloc[-50]
            if trend > 0 and current_price < gap_bottom:  # 上昇トレンドでFVG下
                prob *= 1.3
            elif trend < 0 and current_price > gap_top:  # 下降トレンドでFVG上
                prob *= 0.7

        return min(max(prob, 0), 1)  # 0-1の範囲に制限

# 使用例
if __name__ == "__main__":
    import yfinance as yf

    # データ取得
    symbol = "NVDA"
    stock = yf.Ticker(symbol)
    df = stock.history(period="6mo")

    # 高度な分析
    indicators = AdvancedIndicators()

    # 市場構造分析
    df_structure = indicators.calculate_market_structure(df)
    print(f"最新の市場構造: {df_structure['MarketStructure'].iloc[-1]}")

    # インバランスゾーン検出
    imbalances = indicators.detect_imbalance_zones(df)
    print(f"検出されたインバランス: {len(imbalances)}個")

    # スマートマネー指標
    df['SMF'] = indicators.calculate_smart_money_index(df)
    print(f"最新のスマートマネーフロー: {df['SMF'].iloc[-1]:.2f}")

    # モメンタム指標の計算
    df_with_indicators = indicators.calculate_momentum_indicators(df)
    print(f"最新のRSI: {df_with_indicators['RSI'].iloc[-1]:.2f}")

    # FVG分析
    fvg_analyzer = FVGAnalyzer()
    if imbalances:
        # Find the first bullish FVG for example
        bullish_fvgs = [i for i in imbalances if 'FVG_Bullish' in i['type']]
        if bullish_fvgs:
            fvg = bullish_fvgs[0] # Example: take the first one
            quality = fvg_analyzer.classify_fvg(fvg, df)
            print(f"FVG品質 ({fvg['date'].strftime('%Y-%m-%d')}): {quality}")

            # FVG fill probability example
            if not df.empty:
                current_price_example = df['Close'].iloc[-1]
                fill_prob = fvg_analyzer.calculate_fvg_fill_probability(fvg, current_price_example, df)
                print(f"FVG ({fvg['date'].strftime('%Y-%m-%d')}) が埋まる確率 (現在価格 ${current_price_example:.2f}): {fill_prob:.2%}")
            else:
                print("市場データが空のため、FVG fill probability を計算できません。")

        else:
            print("ブルッシュFVGは見つかりませんでした。")
    else:
        print("インバランスゾーンは見つかりませんでした。")
