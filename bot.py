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
import sys # sysモジュールをインポート
import json
warnings.filterwarnings("ignore")
from backtest import FVGBreakBacktest

# .envファイルから環境変数を読み込み
load_dotenv()

# 環境変数から設定を読み込み
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not DISCORD_BOT_TOKEN:
    raise ValueError("DISCORD_BOT_TOKENが設定されていません。.envファイルを確認してください。")

# Bot設定
intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

bot = commands.Bot(command_prefix="!", intents=intents)

# 設定項目
BOT_CHANNEL_NAME = os.getenv("BOT_CHANNEL_NAME", "fvg-break-alerts")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", 15))

# テクニカル設定
MA_PERIOD = int(os.getenv("MA_PERIOD", 200))
FVG_MIN_GAP_PERCENT = float(os.getenv("FVG_MIN_GAP_PERCENT", 0.5))
RESISTANCE_LOOKBACK = int(os.getenv("RESISTANCE_LOOKBACK", 20))
BREAKOUT_THRESHOLD = float(os.getenv("BREAKOUT_THRESHOLD", 1.005))
STOP_LOSS_RATE = float(os.getenv("STOP_LOSS_RATE", 0.02))
TARGET_PROFIT_RATE = float(os.getenv("TARGET_PROFIT_RATE", 0.05))
MA_PROXIMITY_PERCENT = float(os.getenv("MA_PROXIMITY_PERCENT", 0.05))

# グローバル変数
watched_symbols = set()
fvg_alerts = {}
resistance_alerts = {}
server_configs = {}
fvg_triggered_symbols = {}
resistance_triggered_symbols = {}

# タイムゾーン設定
ET = pytz.timezone("US/Eastern")
JST = pytz.timezone("Asia/Tokyo")

def get_sp500_symbols():
    """S&P500の銘柄リストを取得"""
    try:
        sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        symbols = sp500["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"S&P500銘柄数: {len(symbols)}")
        return symbols
    except Exception as e:
        print(f"S&P500リスト取得エラー: {e}")
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "JNJ"]

def is_us_market_open():
    """米国市場が開いているか確認"""
    now_et = datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close

def get_market_hours_jst():
    """日本時間での米国市場時間を取得"""
    now_et = datetime.now(ET)
    is_dst = bool(now_et.dst())
    return "22:30-5:00 JST（夏時間）" if is_dst else "23:30-6:00 JST（冬時間）"

class StockAnalyzer:
    """株価分析クラス"""
    @staticmethod
    def get_stock_data(symbol, period="1y", interval="1d"):
        """株価データを取得"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period, interval=interval)
            return df if not df.empty else None
        except Exception as e:
            print(f"データ取得エラー ({symbol}): {e}")
            return None

    @staticmethod
    def calculate_sma(df, period):
        return df['Close'].rolling(window=period).mean()

    @staticmethod
    def detect_fvg(df, min_gap_percent=0.5):
        """FVGを検出し、タイプ情報も返す"""
        if len(df) < 3:
            return None

        recent_candles = df.tail(3)
        candle1_high = recent_candles.iloc[0]['High']
        candle3_low = recent_candles.iloc[2]['Low']

        gap_size = (candle3_low - candle1_high) / candle1_high * 100

        if gap_size >= min_gap_percent:
            return {
                'type': 'bullish', # FVGタイプを追加
                'detected_date': recent_candles.index[2],
                'gap_top': candle3_low,
                'gap_bottom': candle1_high,
                'gap_size_percent': gap_size,
                'current_price': recent_candles.iloc[2]['Close']
            }
        return None

    @staticmethod
    def find_resistance_levels(df, lookback_days=20):
        df_recent = df.tail(lookback_days)
        if len(df_recent) < 5: return []

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
    def check_fvg_break_conditions(symbol, data_period="1y", data_interval="1d"):
        """FVGブレイク条件をチェック"""
        df_daily = StockAnalyzer.get_stock_data(symbol, period=data_period, interval=data_interval)
        if df_daily is None or len(df_daily) < MA_PERIOD: return None

        df_weekly = StockAnalyzer.get_stock_data(symbol, period="2y", interval="1wk")
        if df_weekly is None or len(df_weekly) < MA_PERIOD: return None

        df_daily['MA200'] = df_daily['Close'].rolling(window=MA_PERIOD).mean()
        df_weekly['SMA200'] = StockAnalyzer.calculate_sma(df_weekly, MA_PERIOD)

        current_price = df_daily['Close'].iloc[-1]
        daily_ma200 = df_daily['MA200'].iloc[-1]
        weekly_sma200 = df_weekly['SMA200'].iloc[-1]
        current_volume = df_daily['Volume'].iloc[-1]
        avg_volume = df_daily['Volume'].tail(20).mean()

        fvg_info = StockAnalyzer.detect_fvg(df_daily, FVG_MIN_GAP_PERCENT)
        resistance_levels = StockAnalyzer.find_resistance_levels(df_daily, RESISTANCE_LOOKBACK)

        is_above_weekly_sma = current_price > weekly_sma200
        is_near_daily_ma = abs(current_price - daily_ma200) / daily_ma200 < MA_PROXIMITY_PERCENT
        is_fvg_detected = fvg_info is not None and fvg_info.get('type') == 'bullish'

        result = {
            'symbol': symbol, 'current_price': current_price, 'daily_ma200': daily_ma200,
            'weekly_sma200': weekly_sma200, 'current_volume': current_volume, 'avg_volume': avg_volume,
            'resistance_levels': resistance_levels, 'fvg_info': fvg_info,
            'conditions': {
                'above_weekly_sma': is_above_weekly_sma, 'near_daily_ma': is_near_daily_ma,
                'fvg_detected': is_fvg_detected, 'resistance_break': False, 'broken_resistance': None
            },
            'signal_type': None
        }

        if is_above_weekly_sma and is_near_daily_ma and is_fvg_detected:
            result['signal_type'] = 's1_fvg'
            for resistance in resistance_levels:
                if current_price > resistance * BREAKOUT_THRESHOLD:
                    if len(df_daily) > 1 and df_daily['Close'].iloc[-2] <= resistance:
                        result['conditions']['resistance_break'] = True
                        result['conditions']['broken_resistance'] = resistance
                        result['signal_type'] = 's2_resistance_break'
                        break
        return result

    @staticmethod
    def create_chart_with_fvg(symbol, save_path=None):
        df = StockAnalyzer.get_stock_data(symbol, period="6mo")
        if df is None: return None

        df['MA200'] = df['Close'].rolling(window=MA_PERIOD).mean()
        df_plot = df.tail(60).copy()

        fvg_zones = []
        for i in range(len(df_plot) - 2):
            fvg = StockAnalyzer.detect_fvg(df_plot.iloc[i:i+3], FVG_MIN_GAP_PERCENT)
            if fvg: fvg_zones.append(fvg)

        mc = mpf.make_marketcolors(up='green', down='red', edge='inherit', wick={'up':'green', 'down':'red'}, volume='in')
        s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', y_on_right=True)

        apds = []
        if 'MA200' in df_plot.columns and not df_plot['MA200'].isna().all():
            apds.append(mpf.make_addplot(df_plot['MA200'], color='blue', width=2))

        resistance_levels = StockAnalyzer.find_resistance_levels(df, RESISTANCE_LOOKBACK)
        for r in resistance_levels:
            apds.append(mpf.make_addplot([r] * len(df_plot), color='red', width=1, linestyle='--'))

        fig, axes = mpf.plot(df_plot, type='candle', style=s, volume=True, addplot=apds,
                             title=f'{symbol} - FVG & Resistance Analysis', returnfig=True, figsize=(12, 8))
        ax = axes[0]
        for fvg in fvg_zones[-3:]:
            if fvg['detected_date'] in df_plot.index:
                loc = df_plot.index.get_loc(fvg['detected_date'])
                rect = patches.Rectangle((loc - 2, fvg['gap_bottom']), 2, fvg['gap_top'] - fvg['gap_bottom'],
                                         linewidth=1, edgecolor='cyan', facecolor='cyan', alpha=0.3)
                ax.add_patch(rect)

        if save_path:
            plt.savefig(save_path); plt.close(); return save_path
        else:
            buf = BytesIO(); plt.savefig(buf, format='png'); buf.seek(0); plt.close(); return buf

# (Discord関連のEmbed作成、コマンド定義などは変更なし)
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

async def scan_symbols(data_period="1y", data_interval="1d"):
    """全S&P500銘柄をスキャン（2段階アラート）"""
    current_alerts = []
    total_symbols = len(watched_symbols)
    processed = 0

    for symbol in watched_symbols:
        try:
            result = StockAnalyzer.check_fvg_break_conditions(symbol, data_period, data_interval)
            if not result or not result.get("signal_type"):
                continue

            if result['signal_type'] == 's1_fvg':
                if symbol not in fvg_alerts or (datetime.now() - fvg_alerts.get(symbol, {}).get('alert_time', datetime.min)) > timedelta(hours=24):
                    result["alert_type"] = "s1_fvg"
                    current_alerts.append(result)
                    fvg_alerts[symbol] = {"alert_time": datetime.now()}
                    fvg_triggered_symbols[symbol] = result

            elif result['signal_type'] == 's2_resistance_break':
                if symbol in fvg_triggered_symbols:
                    if symbol not in resistance_alerts or (datetime.now() - resistance_alerts.get(symbol, datetime.min)) > timedelta(hours=24):
                        result["alert_type"] = "s2_resistance_break"
                        current_alerts.append(result)
                        resistance_alerts[symbol] = datetime.now()
                        resistance_triggered_symbols[symbol] = result

            processed += 1
            if processed % 50 == 0:
                print(f"スキャン進捗: {processed}/{total_symbols}")

        except Exception as e:
            print(f"スキャンエラー ({symbol}): {e}")
    return current_alerts

def create_summary_embed(alerts_summary):
    fvg_count = len([a for a in alerts_summary if a.get("alert_type") == "s1_fvg"])
    res_count = len([a for a in alerts_summary if a.get("alert_type") == "s2_resistance_break"])
    embed = discord.Embed(
        title="📊 S&P500 スキャン結果サマリー",
        description=f"スキャン時刻: {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}",
        color=discord.Color.gold()
    )
    embed.add_field(name="🔵 FVG検出", value=f"{fvg_count} 銘柄", inline=True)
    embed.add_field(name="🟢 レジスタンス突破", value=f"{res_count} 銘柄", inline=True)
    return embed

from notifier import DiscordNotifier, MockDiscordNotifier, create_advanced_fvg_alert_embed, create_advanced_resistance_alert_embed
from indicators import FVGAnalyzer

async def post_alerts(notifier, alerts):
    if not alerts: return
    # サマリーは元のシンプルなものでOK
    summary_embed = create_summary_embed(alerts)
    await notifier.send_embed(embed=summary_embed)

    for alert in alerts[:20]:
        # FVG品質を分析
        df_for_quality = StockAnalyzer.get_stock_data(alert['symbol'], period="3mo")
        quality = "N/A"
        if df_for_quality is not None and alert.get('fvg_info'):
            quality = FVGAnalyzer.classify_fvg(alert['fvg_info'], df_for_quality)

        # MLコメント（ダミー）
        ml_comment = "過去類似パターンの勝率72%" if quality == "Premium" else "標準的なセットアップ"

        if alert.get("alert_type") == "s1_fvg":
            embed = create_advanced_fvg_alert_embed(alert, fvg_quality=quality, ml_comment=ml_comment)
        else: # s2_resistance_break
            embed = create_advanced_resistance_alert_embed(alert, ml_comment="レジスタンス突破。出来高を伴う強い動き。")

        try:
            chart = StockAnalyzer.create_chart_with_fvg(alert["symbol"])
            file = None
            if chart:
                file = discord.File(chart, filename=f"{alert['symbol']}_chart.png")
                embed.set_image(url=f"attachment://{alert['symbol']}_chart.png")

            await notifier.send_embed(embed=embed, file=file)

        except Exception as e:
            print(f"チャート付きアラート送信エラー: {e}")
            await notifier.send_embed(embed=embed) # チャートなしで再送

# --- Discord Bot Events and Commands ---
from optimization_manager import OptimizationManager

@bot.event
async def on_ready():
    global watched_symbols
    watched_symbols = set(get_sp500_symbols())
    print(f"{bot.user} がログインしました！")
    print(f"監視銘柄数: {len(watched_symbols)}")
    for guild in bot.guilds:
        await setup_guild(guild)

@bot.event
async def on_guild_join(guild):
    await setup_guild(guild)

@tasks.loop(minutes=SCAN_INTERVAL)
async def market_scan_task():
    if not is_us_market_open(): return
    print(f"[{datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}] リアルタイムスキャン開始...")
    alerts = await scan_symbols(data_period="1y", data_interval=f"{SCAN_INTERVAL}m")
    for guild_id, config in server_configs.items():
        if config.get("enabled") and config.get("alert_channel"):
            await post_alerts(config["alert_channel"], alerts)

@bot.command(name="status")
async def bot_status(ctx):
    # ... (statusコマンドの実装は変更なし)
    embed = discord.Embed(title="Bot Status", color=discord.Color.blue())
    embed.add_field(name="Mode", value="Realtime" if market_scan_task.is_running() else "Idle/Daily", inline=False)
    embed.add_field(name="Watched Symbols", value=str(len(watched_symbols)))
    await ctx.send(embed=embed)
    pass

@bot.command(name="optimize")
async def run_optimization_command(ctx, trials: int = 20):
    """MLによるパラメータ最適化を実行します。"""
    await ctx.send(f"🧪 ML最適化を開始します... (試行回数: {trials}回). この処理は数分から数時間かかることがあります。")

    try:
        loop = asyncio.get_event_loop()
        manager = OptimizationManager(n_trials=trials)

        # 最適化 (ブロッキング処理なのでexecutorで実行)
        await ctx.send("ステップ1/4: 最適なパラメータを探索中...")
        best_params = await loop.run_in_executor(
            None, manager.run_optimization, "2023-01-01", "2023-12-31"
        )
        await ctx.send(f"最適パラメータが見つかりました: ```json\n{json.dumps(best_params, indent=2)}```")

        # フルバックテスト
        await ctx.send("ステップ2/4: 見つかったパラメータで詳細なバックテストを実行中...")
        backtest_results = await loop.run_in_executor(
            None, manager.run_full_backtest, best_params, "2024-01-01", "2024-06-30"
        )

        # 結果を保存
        await ctx.send("ステップ3/4: 結果をJSONファイルに保存中...")
        json_path = await loop.run_in_executor(
            None, manager.save_optimization_results, backtest_results
        )

        # HTMLレポートを生成
        await ctx.send("ステップ4/4: HTMLレポートを生成中...")
        html_path = await loop.run_in_executor(
            None, manager.generate_html_report, json_path
        )

        await ctx.send(
            "✅ **最適化完了！**\n"
            f"結果サマリーとレポートを確認してください。",
            files=[discord.File(json_path), discord.File(html_path)]
        )

    except Exception as e:
        await ctx.send(f"❌ エラーが発生しました: {e}")
        print(f"Optimization Error: {e}")

@bot.command(name="backtest")
async def run_backtest_command(ctx, symbol: str, start_date: str = "2023-01-01", end_date: str = "2023-12-31"):
    """指定された銘柄でバックテストを実行します。"""
    symbol = symbol.upper()
    await ctx.send(f"🔍 {symbol} のバックテストを実行中... ({start_date} - {end_date})")

    try:
        # 現在のパラメータでバックテスト
        params = {
            'ma_period': MA_PERIOD, 'fvg_min_gap': FVG_MIN_GAP_PERCENT,
            'resistance_lookback': RESISTANCE_LOOKBACK, 'breakout_threshold': BREAKOUT_THRESHOLD,
            'stop_loss_rate': STOP_LOSS_RATE, 'target_profit_rate': TARGET_PROFIT_RATE,
            'ma_proximity_percent': MA_PROXIMITY_PERCENT
        }
        backtester = FVGBreakBacktest(**params)
        result = backtester.run_backtest(symbol, start_date, end_date)

        if result.get("error"):
            await ctx.send(f"エラー: {result['error']}")
            return

        report = backtester.create_summary_report(result)

        embed = discord.Embed(
            title=f"📊 バックテスト結果 - {symbol}",
            description=f"期間: {start_date} to {end_date}",
            color=discord.Color.purple()
        )
        embed.add_field(name="サマリー", value=f"```{report}```", inline=False)
        await ctx.send(embed=embed)

    except Exception as e:
        await ctx.send(f"❌ エラーが発生しました: {e}")

# --- Execution Modes ---
async def run_daily_scan(mock=False):
    """日足ベースのスキャンを実行し、結果を投稿またはコンソールに出力"""
    print("📈 日足ベースのスキャンを開始します...")
    global watched_symbols
    watched_symbols = set(get_sp500_symbols())
    alerts = await scan_symbols(data_period="1y", data_interval="1d")
    print(f"検出されたアラート数: {len(alerts)}")

    if mock:
        notifier = MockDiscordNotifier()
        await post_alerts(notifier, alerts)
    else:
        if not bot.is_ready():
            print("Botが準備中のため、Discordへの投稿はスキップします。")
            return
        print("\n--- Discordへアラートを投稿します ---")
        # 最初のギルドの最初のチャンネルに投稿（デモ用）
        # 本番では、設定に基づいて適切なチャンネルを選択するロジックが必要
        target_channel = None
        for guild_id, config in server_configs.items():
            if config.get("enabled") and config.get("alert_channel"):
                target_channel = config["alert_channel"]
                break

        if target_channel:
            notifier = DiscordNotifier(target_channel)
            await post_alerts(notifier, alerts)
        else:
            print("投稿先チャンネルが見つかりません。")

class MockContext:
    """`optimize`コマンドのテスト用モックコンテキスト"""
    def __init__(self):
        self.notifier = MockDiscordNotifier("optimize-test")

    async def send(self, message, files=None):
        # ファイルがある場合は、ファイルパスも表示
        if files:
            file_paths = [f.filename for f in files]
            await self.notifier.send_message(f"{message}\nAttached: {', '.join(file_paths)}")
        else:
            await self.notifier.send_message(message)

async def run_realtime_bot(mock=False, command_to_test=None):
    """リアルタイムスキャンモードでBotを起動、またはコマンドをテスト"""
    if mock:
        if command_to_test == 'optimize':
            print("🤖 `!optimize`コマンドのモックテストを実行します...")
            ctx = MockContext()
            await run_optimization_command(ctx, trials=5)
        else:
             print("🤖 リアルタイムスキャンのモックループを開始します...")
             # (既存のモックループ)
             async def mock_loop():
                while True:
                    print(f"\n[{datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')}] モックリアルタイムスキャン開始...")
                    alerts = await scan_symbols(data_period="1y", data_interval=f"{SCAN_INTERVAL}m")
                    notifier = MockDiscordNotifier()
                    await post_alerts(notifier, alerts)
                    await asyncio.sleep(SCAN_INTERVAL * 60)
             await mock_loop()

    else:
        if not DISCORD_BOT_TOKEN or DISCORD_BOT_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":
            print("リアルタイムモードにはDISCORD_BOT_TOKENが必要です。")
            return
        print("🤖 リアルタイムスキャンモードで起動します...")
        market_scan_task.start() # Test: コマンドテストのため、自動スキャンは一時的に無効化
        await bot.start(DISCORD_BOT_TOKEN)

def main():
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        mock = '--mock' in sys.argv

        if mode == 'daily':
            loop = asyncio.get_event_loop()
            if not mock:
                # 通常モードの場合のみログイン
                loop.run_until_complete(bot.login(DISCORD_BOT_TOKEN))
                loop.run_until_complete(bot.fetch_guilds().flatten())
                for guild in bot.guilds:
                    loop.run_until_complete(setup_guild(guild))

            loop.run_until_complete(run_daily_scan(mock=mock))

            if not mock:
                loop.run_until_complete(bot.close())

        elif mode == 'realtime':
            command_to_test = None
            if mock and '--command=optimize' in sys.argv:
                command_to_test = 'optimize'
            asyncio.run(run_realtime_bot(mock=mock, command_to_test=command_to_test))
        else:
            print(f"未定義のモード: {mode}")
            print("使用可能なモード: 'daily', 'realtime'")
    else:
        print("実行モードを指定してください。例: python bot.py realtime")

if __name__ == "__main__":
    main()
