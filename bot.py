import discord
from discord.ext import commands, tasks
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import asyncio
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from io import BytesIO
import warnings
import pytz
warnings.filterwarnings("ignore") # Corrected fancy quote
from backtest import FVGBreakBacktest

# .envファイルから環境変数を読み込み
load_dotenv()

# 環境変数から設定を読み込み
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN") # Corrected fancy quote
if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKENが設定されていません。.envファイルを確認してください。") # Corrected fancy quotes

# Bot設定
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)

# 設定項目
BOT_CHANNEL_NAME = os.getenv("BOT_CHANNEL_NAME", "fvg-break-alerts") # Corrected fancy quotes
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))  # S&P500全体のため15分に延長 # Corrected fancy quotes

# テクニカル設定
MA_PERIOD = int(os.getenv("MA_PERIOD", 200))
FVG_MIN_GAP_PERCENT = float(os.getenv("FVG_MIN_GAP_PERCENT", 0.5))
RESISTANCE_LOOKBACK = int(os.getenv("RESISTANCE_LOOKBACK", 20))
BREAKOUT_THRESHOLD = float(os.getenv("BREAKOUT_THRESHOLD", 1.005))
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", 0.02))
TARGET_PROFIT_RATE = float(os.getenv("TARGET_PROFIT_RATE", 0.05))
MA_PROXIMITY_PERCENT = float(os.getenv("MA_PROXIMITY_PERCENT", 0.05))

# グローバル変数
watched_symbols = set()  # S&P500銘柄で初期化
fvg_alerts = {}  # 銘柄: FVG情報
resistance_alerts = {}  # 銘柄: 最後のレジスタンスアラート時刻
server_configs = {}  # サーバーごとの設定

# アラート発生銘柄のリスト
fvg_triggered_symbols = {}  # 第一アラート発生銘柄
resistance_triggered_symbols = {}  # 第二アラート発生銘柄

# タイムゾーン設定
ET = pytz.timezone("US/Eastern") # Corrected fancy quote
JST = pytz.timezone("Asia/Tokyo") # Corrected fancy quote

def get_sp500_symbols():
    """S&P500の銘柄リストを取得""" # Corrected fancy quote
    try:
        # S&P500の銘柄リストを取得
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0] # Corrected fancy quote
        symbols = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()  # BRK.BをBRK-Bに変換 # Corrected fancy quotes and added regex=False
        print(f"S&P500銘柄数: {len(symbols)}") # Corrected fancy quote
        return symbols
    except Exception as e:
        print(f"S&P500リスト取得エラー: {e}") # Corrected fancy quote
        # フォールバック：主要銘柄のみ
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ"] # Corrected fancy quotes

def is_us_market_open():
    """米国市場が開いているか確認（夏時間・冬時間対応）""" # Corrected fancy quote
    now_et = datetime.now(ET)

    # 週末チェック
    if now_et.weekday() >= 5:  # 土日
        return False

    # 米国市場時間（9:30-16:00 ET）
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_et <= market_close

def get_market_hours_jst():
    """日本時間での米国市場時間を取得""" # Corrected fancy quote
    now_et = datetime.now(ET)

    # 夏時間チェック（3月第2日曜日～11月第1日曜日）
    is_dst = bool(now_et.dst())

    if is_dst:
        return "22:30-5:00 JST（夏時間）"
    else:
        return "23:30-6:00 JST（冬時間）"

class StockAnalyzer:
    """株価分析クラス - FVG検出機能付き""" # Corrected fancy quote

    @staticmethod
    def get_stock_data(symbol, period="1y"):
        """株価データを取得"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if df.empty:
                return None
            return df
        except Exception as e:
            print(f"データ取得エラー ({symbol}): {e}")
            return None

    @staticmethod
    def calculate_sma(df, period):
        """単純移動平均（SMA）を計算"""
        return df['Close'].rolling(window=period).mean()

    @staticmethod
    def detect_fvg(df, min_gap_percent=0.5):
        """FVG（Fair Value Gap）を検出"""
        if len(df) < 3:
            return None

        # 最新3本のローソク足を取得
        recent_candles = df.tail(3)

        # FVG条件: 3本目の安値 > 1本目の高値
        candle1_high = recent_candles.iloc[0]['High']
        candle3_low = recent_candles.iloc[2]['Low']

        gap_size = (candle3_low - candle1_high) / candle1_high * 100

        if gap_size >= min_gap_percent:
            return {
                'detected_date': recent_candles.index[2],
                'gap_top': candle3_low,
                'gap_bottom': candle1_high,
                'gap_size_percent': gap_size,
                'candle2_close': recent_candles.iloc[1]['Close'],
                'current_price': recent_candles.iloc[2]['Close']
            }

        return None

    @staticmethod
    def find_resistance_levels(df, lookback_days=20):
        """直近の高値（レジスタンス）を検出"""
        df_recent = df.tail(lookback_days)
        if len(df_recent) < 5:
            return []

        recent_high = df_recent['High'].max()

        highs = []
        for i in range(2, len(df_recent) - 2):
            if (df_recent['High'].iloc[i] > df_recent['High'].iloc[i-1] and
                df_recent['High'].iloc[i] > df_recent['High'].iloc[i-2] and
                df_recent['High'].iloc[i] > df_recent['High'].iloc[i+1] and
                df_recent['High'].iloc[i] > df_recent['High'].iloc[i+2]):
                highs.append(df_recent['High'].iloc[i])

        all_highs = [recent_high] + highs
        unique_highs = []
        for high in sorted(all_highs, reverse=True):
            if not unique_highs or all(abs(high - h) / h > 0.01 for h in unique_highs):
                unique_highs.append(high)

        return unique_highs[:3]

    @staticmethod
    def check_fvg_break_conditions(symbol):
        """FVGブレイク条件をチェック（2段階）"""
        # 日足データ取得
        df_daily = StockAnalyzer.get_stock_data(symbol, period="1y")
        if df_daily is None or len(df_daily) < MA_PERIOD:
            return None

        # 週足データ取得
        stock = yf.Ticker(symbol)
        df_weekly = stock.history(period="2y", interval="1wk")
        if df_weekly is None or len(df_weekly) < MA_PERIOD:
            return None

        # 移動平均計算
        df_daily['MA200'] = df_daily['Close'].rolling(window=MA_PERIOD).mean()
        df_weekly['SMA200'] = StockAnalyzer.calculate_sma(df_weekly, MA_PERIOD)

        # 最新データ
        current_price = df_daily['Close'].iloc[-1]
        daily_ma200 = df_daily['MA200'].iloc[-1]
        weekly_sma200 = df_weekly['SMA200'].iloc[-1]
        current_volume = df_daily['Volume'].iloc[-1]
        avg_volume = df_daily['Volume'].tail(20).mean()

        # FVG検出
        fvg_info = StockAnalyzer.detect_fvg(df_daily, FVG_MIN_GAP_PERCENT)

        # レジスタンスレベル検出
        resistance_levels = StockAnalyzer.find_resistance_levels(df_daily, RESISTANCE_LOOKBACK)

        # 基本条件チェック
        ma_distance = abs(current_price - daily_ma200) / daily_ma200
        is_above_weekly_sma = current_price > weekly_sma200
        is_near_daily_ma = ma_distance < MA_PROXIMITY_PERCENT
        is_fvg_detected = fvg_info is not None and fvg_info['type'] == 'bullish'

        result = {
            'symbol': symbol,
            'current_price': current_price,
            'daily_ma200': daily_ma200,
            'weekly_sma200': weekly_sma200,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'resistance_levels': resistance_levels,
            'fvg_info': fvg_info,
            'conditions': {
                'above_weekly_sma': is_above_weekly_sma,
                'near_daily_ma': is_near_daily_ma,
                'fvg_detected': is_fvg_detected,
                'resistance_break': False,
                'broken_resistance': None
            },
            'signal_type': None
        }

        # シグナルタイプを判定（2段階戦略）
        if is_above_weekly_sma and is_near_daily_ma and is_fvg_detected:
            # 戦略1の条件成立
            result['signal_type'] = 's1_fvg'

            # 戦略2の条件（レジスタンス突破）をチェック
            for resistance in resistance_levels:
                if current_price > resistance * BREAKOUT_THRESHOLD:
                    if len(df_daily) > 1 and df_daily['Close'].iloc[-2] <= resistance:
                        result['conditions']['resistance_break'] = True
                        result['conditions']['broken_resistance'] = resistance
                        result['signal_type'] = 's2_resistance_break' # S2が成立すれば上書き
                        break

        return result

    @staticmethod
    def create_chart_with_fvg(symbol, save_path=None):
        """FVGゾーンを含むチャートを作成"""
        df = StockAnalyzer.get_stock_data(symbol, period="6mo")
        if df is None:
            return None

        df['MA200'] = df['Close'].rolling(window=MA_PERIOD).mean()
        df_plot = df.tail(60).copy()

        # FVG検出
        fvg_zones = []
        for i in range(len(df_plot) - 2):
            fvg = StockAnalyzer.detect_fvg(df_plot.iloc[i:i+3], FVG_MIN_GAP_PERCENT)
            if fvg:
                fvg_zones.append(fvg)

        # カスタムスタイル
        mc = mpf.make_marketcolors(
            up='green', down='red',
            edge='inherit',
            wick={'up': 'green', 'down': 'red'},
            volume='in'
        )
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)

        # プロット設定
        apds = []
        if 'MA200' in df_plot.columns and not df_plot['MA200'].isna().all():
            apds.append(mpf.make_addplot(df_plot['MA200'], color='blue', width=2))

        # レジスタンスライン
        resistance_levels = StockAnalyzer.find_resistance_levels(df, RESISTANCE_LOOKBACK)
        for resistance in resistance_levels[:3]:
            apds.append(mpf.make_addplot([resistance] * len(df_plot), color='red', width=1, linestyle='--'))

        # チャート作成
        fig, axes = mpf.plot(
            df_plot,
            type='candle',
            style=s,
            volume=True,
            addplot=apds if apds else None,
            title=f'{symbol} - FVG & Resistance Analysis',
            returnfig=True,
            figsize=(12, 8)
        )

        # FVGゾーンを描画
        ax = axes[0]
        for fvg in fvg_zones[-3:]: # Plotting only last 3 FVGs for clarity
            # Ensure detected_date is in df_plot.index before trying to get its location
            if fvg['detected_date'] in df_plot.index:
                fvg_x_location = df_plot.index.get_loc(fvg['detected_date'])
                rect_x = fvg_x_location - 2 # Adjust x-position for visibility
                if rect_x < 0: rect_x = 0 # Ensure not less than 0
            else:
                # Fallback or skip if date not found, this might happen with very short df_plot
                # Fallback or skip if date not found, this might happen with very short df_plot
                # For simplicity, let's try to place it relative to the end if date is missing
                # This part might need more robust handling depending on data scenarios
                # rect_x = len(df_plot) - 5 # Removed this fallback as it can cause issues
                continue # Skip this FVG if date is not found


            rect = patches.Rectangle(
                (rect_x, fvg['gap_bottom']),
                2, # Width of FVG rectangle
                fvg['gap_top'] - fvg['gap_bottom'],
                linewidth=1,
                edgecolor='cyan',
                facecolor='cyan',
                alpha=0.3
            )
            ax.add_patch(rect)

        if save_path:
            plt.savefig(save_path)
            plt.close()
            return save_path
        else:
            buf = BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()
            return buf

async def setup_guild(guild):
    """サーバーの初期設定"""
    alert_channel = None
    for channel in guild.text_channels:
        if channel.name == BOT_CHANNEL_NAME:
            alert_channel = channel
            break

    if not alert_channel:
        try:
            alert_channel = await guild.create_text_channel(
                name=BOT_CHANNEL_NAME,
                topic="📈 FVG Break Alert - S&P500 Fair Value Gap & レジスタンス突破アラート"
            )
        except discord.Forbidden:
            print(f"チャンネル作成権限がありません: {guild.name}")

    server_configs[guild.id] = {
        "alert_channel": alert_channel,
        "enabled": True
    }

    if alert_channel:
        print(f"サーバー '{guild.name}' の設定完了。アラートチャンネル: #{alert_channel.name}")

def create_fvg_alert_embed(result):
    """戦略1（FVG検出）のアラート用Embed"""
    symbol = result["symbol"]
    fvg = result["fvg_info"]
    company_name = yf.Ticker(symbol).info.get("longName", symbol)

    embed = discord.Embed(
        title=f"📈 戦略1: FVG検出 - {symbol}",
        description=f"**{company_name}** がエントリー条件を満たし、FVGを検出しました。",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )
    embed.add_field(
        name="📊 FVG情報",
        value=f"• 上限: `${fvg['gap_top']:.2f}`\n• 下限: `${fvg['gap_bottom']:.2f}`\n• サイズ: `{fvg['gap_size_percent']:.2f}%`",
        inline=False
    )
    embed.add_field(
        name="環境",
        value=f"価格: `${result['current_price']:.2f}`\n週足SMA: `${result['weekly_sma200']:.2f}`\n日足MA: `${result['daily_ma200']:.2f}`",
        inline=False
    )
    first_resistance = result["resistance_levels"][0] if result["resistance_levels"] else "N/A"
    embed.add_field(
        name="次のステップ",
        value=f"次のレジスタンス `{first_resistance:.2f}` の突破を監視します (戦略2)。",
        inline=False
    )
    return embed

def create_resistance_alert_embed(result):
    """戦略2（レジスタンス突破）のアラート用Embed"""
    symbol = result["symbol"]
    resistance = result["conditions"]["broken_resistance"]
    current_price = result["current_price"]
    company_name = yf.Ticker(symbol).info.get("longName", symbol)
    price_change = ((current_price - resistance) / resistance) * 100

    embed = discord.Embed(
        title=f"🚀 戦略2: レジスタンス突破 - {symbol}",
        description=f"**{company_name}** が戦略1の条件達成後、レジスタンスを突破しました！",
        color=discord.Color.green(),
        timestamp=datetime.now()
    )
    embed.add_field(
        name="💥 ブレイク情報",
        value=f"• 現在価格: `${current_price:.2f}`\n• 突破ライン: `${resistance:.2f}` (+{price_change:.1f}%)",
        inline=False
    )
    volume_ratio = result["current_volume"] / result["avg_volume"]
    embed.add_field(
        name="テクニカル",
        value=f"• FVG発生済み: ✅\n• 出来高: 通常の`{volume_ratio:.1f}`倍",
        inline=False
    )
    target_price = current_price * (1 + TARGET_PROFIT_RATE)
    stop_loss_price = result['fvg_info']['gap_bottom'] * (1 - STOP_LOSS_RATE)
    embed.add_field(
        name="トレード戦略（例）",
        value=f"• 目標利益: `${target_price:.2f}`\n• ストップロス: `${stop_loss_price:.2f}`",
        inline=False
    )
    return embed

async def scan_symbols():
    """全S&P500銘柄をスキャン（2段階アラート）"""
    current_alerts = []
    total_symbols = len(watched_symbols)
    processed = 0

    for symbol in watched_symbols:
        try:
            result = StockAnalyzer.check_fvg_break_conditions(symbol)
            if not result or not result.get("signal_type"):
                continue

            # 戦略1: FVG検出アラート
            if result['signal_type'] == 's1_fvg':
                # 24時間に1回のみアラート
                if symbol not in fvg_alerts or (datetime.now() - fvg_alerts.get(symbol, {}).get('alert_time', datetime.min)) > timedelta(hours=24):
                    result["alert_type"] = "s1_fvg"
                    current_alerts.append(result)
                    fvg_alerts[symbol] = {"alert_time": datetime.now()}
                    fvg_triggered_symbols[symbol] = result # リスト更新

            # 戦略2: レジスタンス突破アラート
            elif result['signal_type'] == 's2_resistance_break':
                # S1アラートが出ていることが前提
                if symbol in fvg_triggered_symbols:
                    # 24時間に1回のみアラート
                    if symbol not in resistance_alerts or (datetime.now() - resistance_alerts.get(symbol, datetime.min)) > timedelta(hours=24):
                        result["alert_type"] = "s2_resistance_break"
                        current_alerts.append(result)
                        resistance_alerts[symbol] = datetime.now()
                        resistance_triggered_symbols[symbol] = result # リスト更新
                        # S2が出たらS1リストからは削除しても良い（重複を避けるため）
                        # del fvg_triggered_symbols[symbol]

            processed += 1
            if processed % 50 == 0:
                print(f"スキャン進捗: {processed}/{total_symbols}")

        except Exception as e:
            print(f"スキャンエラー ({symbol}): {e}")
    return current_alerts

def create_summary_embed(alerts_summary): # Renamed parameter
    """アラートサマリーのEmbed作成"""
    fvg_count = len([a for a in alerts_summary if a.get("alert_type") == "fvg"])
    resistance_count = len([a for a in alerts_summary if a.get("alert_type") == "resistance"])

    embed = discord.Embed(
        title="📊 S&P500 スキャン結果サマリー",
        description=f"スキャン時刻: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}",
        color=discord.Color.gold(),
        timestamp=datetime.now()
    )

    embed.add_field(
        name="🔵 FVG検出（第1アラート）",
        value=f"{fvg_count} 銘柄",
        inline=True
    )

    embed.add_field(
        name="🟢 レジスタンス突破（第2アラート）",
        value=f"{resistance_count} 銘柄",
        inline=True
    )

    # FVG銘柄リスト（最大10銘柄）
    if fvg_count > 0:
        fvg_symbols = [a["symbol"] for a in alerts_summary if a.get("alert_type") == "fvg"][:10]
        fvg_list_str = ", ".join(fvg_symbols) # Renamed variable
        if fvg_count > 10:
            fvg_list_str += f" 他{fvg_count - 10}銘柄"
        embed.add_field(
            name="FVG発生銘柄",
            value=fvg_list_str,
            inline=False
        )

    # レジスタンス突破銘柄リスト（最大10銘柄）
    if resistance_count > 0:
        resistance_symbols = [a["symbol"] for a in alerts_summary if a.get("alert_type") == "resistance"][:10]
        resistance_list_str = ", ".join(resistance_symbols) # Renamed variable
        if resistance_count > 10:
            resistance_list_str += f" 他{resistance_count - 10}銘柄"
        embed.add_field(
            name="レジスタンス突破銘柄",
            value=resistance_list_str,
            inline=False
        )
    return embed

@bot.event
async def on_ready():
    bot.start_time = datetime.now()
    print(f"{bot.user} がログインしました！")

    # S&P500銘柄を取得
    global watched_symbols
    watched_symbols = set(get_sp500_symbols())

    print(f"監視銘柄数: {len(watched_symbols)}")
    print(f"スキャン間隔: {SCAN_INTERVAL}分")
    print("戦略: S&P500 2段階アラート（FVG + レジスタンス突破）")
    print(f"米国市場時間: {get_market_hours_jst()}")

    # 既存サーバーの設定
    for guild in bot.guilds:
        await setup_guild(guild)

    # スキャンタスク開始
    if not market_scan_task.is_running():
        market_scan_task.start()

@bot.event
async def on_guild_join(guild):
    """新しいサーバーに参加した時"""
    await setup_guild(guild)

@tasks.loop(minutes=SCAN_INTERVAL)
async def market_scan_task():
    """定期的な市場スキャン"""
    # 米国市場時間チェック
    if not is_us_market_open():
        return

    now_et = datetime.now(ET)
    print(f"[{now_et.strftime('%Y-%m-%d %H:%M ET')}] S&P500スキャン開始...")

    current_scan_alerts = await scan_symbols() # Renamed variable

    if current_scan_alerts:
        for guild_id, config in server_configs.items():
            if config.get("enabled") and config.get("alert_channel"): # Use .get for safety
                # サマリーを最初に送信
                summary_embed = create_summary_embed(current_scan_alerts)
                await config["alert_channel"].send(embed=summary_embed)
                
                # 個別アラートは最大20件まで
                for alert_item in current_scan_alerts[:20]:
                    if alert_item.get("alert_type") == "s1_fvg":
                        embed = create_fvg_alert_embed(alert_item)
                    elif alert_item.get("alert_type") == "s2_resistance_break":
                        embed = create_resistance_alert_embed(alert_item)
                    else:
                        continue

                    try:
                        chart_buffer = StockAnalyzer.create_chart_with_fvg(alert_item["symbol"])
                        if chart_buffer:
                            file = discord.File(chart_buffer, filename=f"{alert_item['symbol']}_fvg_chart.png")
                            embed.set_image(url=f"attachment://{alert_item['symbol']}_fvg_chart.png")
                            await config["alert_channel"].send(embed=embed, file=file)
                        else:
                            await config["alert_channel"].send(embed=embed)
                    except Exception as e:
                        print(f"チャート送信エラー: {e}")
                        await config["alert_channel"].send(embed=embed)

@bot.command(name="fvg_list")
async def show_fvg_list(ctx):
    """第一アラート（FVG）発生銘柄のリストを表示"""
    if not fvg_triggered_symbols:
        await ctx.send("現在、FVGアラートが発生している銘柄はありません。")
        return

    embed = discord.Embed(
        title="🔵 FVG発生銘柄リスト",
        description=f"合計 {len(fvg_triggered_symbols)} 銘柄",
        color=discord.Color.blue(),
        timestamp=datetime.now()
    )

    # 銘柄を価格帯別に分類
    symbols_by_price = {}
    for symbol, data in fvg_triggered_symbols.items():
        price = data.get("current_price") # Use .get for safety
        if price is None: continue # Skip if price is not available

        if price < 50:
            key = "$0-50"
        elif price < 100:
            key = "$50-100"
        elif price < 200:
            key = "$100-200"
        else:
            key = "$200+"

        if key not in symbols_by_price:
            symbols_by_price[key] = []
        symbols_by_price[key].append(f"{symbol} (${price:.2f})")

    for price_range in sorted(symbols_by_price.keys()):
        symbols_list = symbols_by_price[price_range] # Renamed variable
        embed.add_field(
            name=price_range,
            value="\n".join(symbols_list[:10]),  # 最大10銘柄
            inline=True
        )
    await ctx.send(embed=embed)

@bot.command(name="resistance_list")
async def show_resistance_list(ctx):
    """第二アラート（レジスタンス突破）発生銘柄のリストを表示"""
    if not resistance_triggered_symbols:
        await ctx.send("現在、レジスタンス突破アラートが発生している銘柄はありません。")
        return

    embed = discord.Embed(
        title="🟢 レジスタンス突破銘柄リスト",
        description=f"合計 {len(resistance_triggered_symbols)} 銘柄",
        color=discord.Color.green(),
        timestamp=datetime.now()
    )

    # 上昇率順にソート
    sorted_symbols_list = [] # Renamed variable
    for symbol, data in resistance_triggered_symbols.items():
        resistance = data.get("conditions", {}).get("broken_resistance") # Use .get for safety
        current_price = data.get("current_price") # Use .get for safety
        if resistance is None or current_price is None: continue # Skip if data is incomplete

        change_pct = ((current_price - resistance) / resistance) * 100
        sorted_symbols_list.append((symbol, current_price, change_pct))

    sorted_symbols_list.sort(key=lambda x: x[2], reverse=True)

    # 上位20銘柄を表示
    top_symbols_list = [] # Renamed variable
    for symbol, price, change in sorted_symbols_list[:20]:
        top_symbols_list.append(f"{symbol}: ${price:.2f} (+{change:.1f}%)")

    embed.add_field(
        name="上昇率上位",
        value="\n".join(top_symbols_list[:10]),
        inline=False
    )

    if len(top_symbols_list) > 10:
        embed.add_field(
            name="続き",
            value="\n".join(top_symbols_list[10:20]),
            inline=False
        )
    await ctx.send(embed=embed)

@bot.command(name="check")
async def check_symbol(ctx, symbol_arg: str): # Renamed argument
    """特定銘柄の現在状態とチェック"""
    symbol_to_check = symbol_arg.upper() # Renamed variable
    await ctx.send(f"🔍 {symbol_to_check} を分析中...")

    result = StockAnalyzer.check_fvg_break_conditions(symbol_to_check)
    if not result:
        await ctx.send(f"❌ {symbol_to_check} のデータを取得できません。")
        return

    # 企業情報を取得
    try:
        ticker = yf.Ticker(symbol_to_check)
        company_name = ticker.info.get("longName", symbol_to_check)
    except:
        company_name = symbol_to_check

    color = discord.Color.green() if result.get("signal_type") else discord.Color.blue()
    embed = discord.Embed(
        title=f"📊 {symbol_to_check} - {company_name}",
        color=color
    )

    embed.add_field(
        name="💰 現在価格",
        value=f"${result.get('current_price', 0):.2f}", # Use .get with default
        inline=True
    )

    conditions_text = []
    conditions_text.append(f"{'✅' if result.get('conditions', {}).get('above_weekly_sma') else '❌'} 週足200SMA以上")
    conditions_text.append(f"{'✅' if result.get('conditions', {}).get('near_daily_ma') else '❌'} 日足200MA付近")

    embed.add_field(
        name="📋 基本条件",
        value="\n".join(conditions_text),
        inline=False
    )

    if result.get("fvg_info"):
        fvg = result["fvg_info"]
        embed.add_field(
            name="🔵 FVG検出",
            value=f"ギャップ: ${fvg.get('gap_bottom', 0):.2f} - ${fvg.get('gap_top', 0):.2f}\n" # Use .get with default
                  f"サイズ: {fvg.get('gap_size_percent', 0):.2f}%",
            inline=True
        )

    if result.get("resistance_levels"):
        resistance_text = "\n".join([f"• ${r:.2f}" for r in result.get("resistance_levels", [])[:3]]) # Use .get with default
        embed.add_field(
            name="🎯 レジスタンスレベル",
            value=resistance_text,
            inline=True
        )

    # チャート添付
    try:
        chart_buffer = StockAnalyzer.create_chart_with_fvg(symbol_to_check)
        if chart_buffer:
            file = discord.File(chart_buffer, filename=f"{symbol_to_check}_analysis.png")
            embed.set_image(url=f"attachment://{symbol_to_check}_analysis.png")
            await ctx.send(embed=embed, file=file)
            return
    except Exception as e:
        print(f"チャート作成エラー: {e}")

    await ctx.send(embed=embed)

@bot.command(name="status")
async def bot_status(ctx):
    """ボットの状態を表示"""
    embed = discord.Embed(
        title="🤖 S&P500 FVG Break Alert Bot ステータス",
        color=discord.Color.blue()
    )
    embed.add_field(
        name="監視銘柄数",
        value=f"{len(watched_symbols)} 銘柄 (S&P500)",
        inline=True
    )

    embed.add_field(
        name="スキャン間隔",
        value=f"{SCAN_INTERVAL} 分",
        inline=True
    )

    # 市場状態
    market_status_text = "🟢 開場中" if is_us_market_open() else "🔴 閉場中" # Renamed variable
    embed.add_field(
        name="米国市場",
        value=f"{market_status_text}\n{get_market_hours_jst()}",
        inline=True
    )

    if hasattr(bot, "start_time"):
        uptime = datetime.now() - bot.start_time
        embed.add_field(
            name="稼働時間",
            value=f"{uptime.days}日 {uptime.seconds//3600}時間",
            inline=True
        )

    embed.add_field(
        name="🔵 FVG発生銘柄",
        value=f"{len(fvg_triggered_symbols)} 銘柄",
        inline=True
    )

    embed.add_field(
        name="🟢 突破銘柄",
        value=f"{len(resistance_triggered_symbols)} 銘柄",
        inline=True
    )

    # 最新のアラート
    all_alerts_data = [] # Renamed to avoid conflict
    for symbol, fvg_data in fvg_alerts.items():
        all_alerts_data.append((symbol, fvg_data.get("alert_time", datetime.min), "FVG")) # Use .get for safety
    for symbol, alert_time in resistance_alerts.items():
        all_alerts_data.append((symbol, alert_time if isinstance(alert_time, datetime) else datetime.min, "Resistance")) # Ensure alert_time is datetime

    recent_alerts_data = sorted(all_alerts_data, key=lambda x: x[1], reverse=True)[:5] # Renamed variable
    if recent_alerts_data:
        alert_text_list = [f"{s}: {at}" for s, t, at in recent_alerts_data] # Renamed variable
        embed.add_field(
            name="最近のアラート",
            value="\n".join(alert_text_list),
            inline=False
        )
    await ctx.send(embed=embed)

@bot.command(name="scan")
async def manual_scan(ctx):
    """手動スキャン実行（管理者のみ）"""
    if not ctx.author.guild_permissions.administrator:
        await ctx.send("このコマンドは管理者のみ使用できます。")
        return

    await ctx.send("🔍 S&P500全銘柄の手動スキャンを開始します... (数分かかります)")

    manual_scan_alerts = await scan_symbols() # Renamed variable

    if manual_scan_alerts:
        summary_embed = create_summary_embed(manual_scan_alerts)
        await ctx.send(embed=summary_embed)
    else:
        await ctx.send("シグナルは検出されませんでした。")

# Botを起動
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
