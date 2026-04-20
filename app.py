import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# 1. ページ設定
st.set_page_config(page_title="FX リアルタイム予測", layout="wide")

# 2. 自動更新の設定（例：60,000ミリ秒 = 1分ごとにページをリフレッシュ）
# これにより、常に最新のデータを取得しに行きます
st_autorefresh(interval=60 * 1000, key="fxtracker")

st.title("💹 ドル円 リアルタイム・トレンド予測")
st.caption("1分ごとに自動更新中（yfinanceのデータ仕様により数分の遅延が含まれる場合があります）")

# 3. データ取得関数の改良
@st.cache_data(ttl=60) # キャッシュを10分から「1分(60秒)」に短縮
def get_latest_data():
    # 直近1日分の1分足を取得（最新の状態を確保するため）
    df = yf.download("USDJPY=X", period="1d", interval="1m")
    if df.empty:
        return pd.DataFrame()
    
    # 最新の10分足にリサンプリング
    df_10m = df['Close'].resample('10min').last().dropna().reset_index()
    df_10m.columns = ['ds', 'y']
    df_10m['ds'] = df_10m['ds'].dt.tz_localize(None)
    return df_10m

try:
    df = get_latest_data()

    if len(df) < 2:
        st.error("データの取得に失敗しました。市場が閉まっているか、一時的な通信エラーです。")
    else:
        # 最新価格の表示
        current_price = df['y'].iloc[-1]
        last_update = df['ds'].iloc[-1].strftime('%H:%M:%S')
        
        st.subheader(f"現在価格: {current_price:.3f} JPY (更新: {last_update})")

        # 予測ロジック
        model = Prophet(
            changepoint_prior_scale=0.01,
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False
        )
        model.fit(df)
        
        future = model.make_future_dataframe(periods=3, freq='10min')
        forecast = model.predict(future)
        
        # グラフ表示
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="実績", line=dict(color='#00CF80', width=2)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="予測", line=dict(dash='dash', color='#FF4B4B')))
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=400,
            template="plotly_dark", # 短期トレードで見やすいダークモード
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # 予測値の提示
        next_pred = forecast['yhat'].iloc[-1]
        st.info(f"次回の10分足 予測ターゲット: {next_pred:.3f} JPY")

except Exception as e:
    st.info("データ更新待ち、または市場休止中です。")
