import discord
from datetime import datetime
import yfinance as yf
from io import BytesIO

class DiscordNotifier:
    """Discordへの通知を管理するクラス"""
    def __init__(self, channel):
        self.channel = channel

    async def send_embed(self, embed, file=None):
        if not self.channel:
            print("通知チャンネルが設定されていません。")
            return
        await self.channel.send(embed=embed, file=file)

    async def send_message(self, message):
        if not self.channel:
            print("通知チャンネルが設定されていません。")
            return
        await self.channel.send(message)

    async def send_file(self, file_path, message=""):
        if not self.channel:
            print("通知チャンネルが設定されていません。")
            return
        await self.channel.send(message, file=discord.File(file_path))


class MockDiscordNotifier:
    """Discord通知をコンソールに出力するモッククラス"""
    def __init__(self, channel_name="mock-channel"):
        self.channel_name = channel_name
        print(f"--- MockDiscordNotifier initialized for channel: #{self.channel_name} ---")

    async def send_embed(self, embed, file=None):
        print(f"\n--- [Mock Embed Send to #{self.channel_name}] ---")
        print(f"Title: {embed.title}")
        print(f"Description: {embed.description}")
        for field in embed.fields:
            print(f"  - {field.name}: {field.value.replace('`', '')}") # Clean up backticks for readability
        if file:
            print(f"  - Attached File: {file.filename}")
        print("------------------------------------------")

    async def send_message(self, message):
        print(f"\n--- [Mock Message Send to #{self.channel_name}] ---")
        print(message)
        print("------------------------------------------")

    async def send_file(self, file_path, message=""):
        print(f"\n--- [Mock File Send to #{self.channel_name}] ---")
        print(f"Message: {message}")
        print(f"File: {file_path}")
        print("------------------------------------------")


def create_advanced_fvg_alert_embed(result, fvg_quality="Standard", ml_comment="N/A"):
    """[高度化版] 戦略1（FVG検出）のアラート用Embed"""
    symbol = result["symbol"]
    fvg = result["fvg_info"]
    try:
        company_name = yf.Ticker(symbol).info.get("longName", symbol)
    except:
        company_name = symbol

    embed = discord.Embed(
        title=f"🔵 FVG検出アラート - {symbol}",
        description=f"**{company_name}**",
        color=0x3498db, # Blue
        timestamp=datetime.now()
    )
    embed.set_footer(text="FVG Break Alert Bot | Strategy 1")

    embed.add_field(
        name="現在価格",
        value=f"${result['current_price']:.2f}",
        inline=True
    )
    embed.add_field(
        name="FVG品質",
        value=f"**{fvg_quality}**",
        inline=True
    )
    embed.add_field(
        name="出来高比",
        value=f"{result['current_volume'] / result['avg_volume']:.1f}倍",
        inline=True
    )

    embed.add_field(
        name="📊 FVGゾーン",
        value=f"`${fvg['gap_bottom']:.2f}` - `${fvg['gap_top']:.2f}` ({fvg['gap_size_percent']:.2f}%)",
        inline=False
    )

    embed.add_field(
        name="📈 テクニカル状況",
        value=f"• 週足SMA200: {'✅ 上昇' if result['conditions']['above_weekly_sma'] else '❌ 下降'}\n"
              f"• 日足MA20: {'✅ 付近' if result['conditions']['near_daily_ma'] else '❌ 乖離'}",
        inline=False
    )

    first_resistance = result["resistance_levels"][0] if result["resistance_levels"] else None
    if first_resistance:
        stop_loss = fvg['gap_bottom'] * (1 - result.get('stop_loss_rate', 0.02))
        target_profit = first_resistance * 1.05 # Example target
        risk_reward = (target_profit - result['current_price']) / (result['current_price'] - stop_loss) if (result['current_price'] - stop_loss) > 0 else 0

        embed.add_field(
            name="🎯 トレードプラン (例)",
            value=f"• 次の関門: `${first_resistance:.2f}`\n"
                  f"• ストップロス: `${stop_loss:.2f}`\n"
                  f"• リスクリワード比: `1 : {risk_reward:.1f}`",
            inline=False
        )

    if ml_comment != "N/A":
        embed.add_field(
            name="💬 ML最適化コメント",
            value=f"_{ml_comment}_",
            inline=False
        )

    return embed

def create_advanced_resistance_alert_embed(result, ml_comment="N/A"):
    """[高度化版] 戦略2（レジスタンス突破）のアラート用Embed"""
    symbol = result["symbol"]
    resistance = result["conditions"]["broken_resistance"]
    current_price = result["current_price"]
    try:
        company_name = yf.Ticker(symbol).info.get("longName", symbol)
    except:
        company_name = symbol

    embed = discord.Embed(
        title=f"🟢 レジスタンス突破アラート - {symbol}",
        description=f"**{company_name}**",
        color=0x2ecc71, # Green
        timestamp=datetime.now()
    )
    embed.set_footer(text="FVG Break Alert Bot | Strategy 2")

    embed.add_field(
        name="現在価格",
        value=f"${current_price:.2f}",
        inline=True
    )
    embed.add_field(
        name="突破ライン",
        value=f"`${resistance:.2f}`",
        inline=True
    )

    # FVGからの経過日数を計算
    fvg_date = pd.to_datetime(result['fvg_info']['detected_date']).tz_localize(None)
    current_date = pd.to_datetime(datetime.now().date())
    days_since_fvg = (current_date - fvg_date).days

    embed.add_field(
        name="FVGからの日数",
        value=f"{days_since_fvg}日",
        inline=True
    )

    embed.add_field(
        name="⚡ アクション",
        value="推奨: **エントリー実行**",
        inline=False
    )

    stop_loss = result['fvg_info']['gap_bottom'] * (1 - result.get('stop_loss_rate', 0.02))
    target_profit = current_price * (1 + result.get('target_profit_rate', 0.05))
    risk_reward = (target_profit - current_price) / (current_price - stop_loss) if (current_price - stop_loss) > 0 else 0

    embed.add_field(
        name="📈 トレード戦略",
        value=f"• 目標価格: `${target_profit:.2f}`\n"
              f"• ストップロス: `${stop_loss:.2f}`\n"
              f"• リスクリワード比: `1 : {risk_reward:.1f}`",
        inline=False
    )

    if ml_comment != "N/A":
        embed.add_field(
            name="💬 ML最適化コメント",
            value=f"_{ml_comment}_",
            inline=False
        )

    return embed
