import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import time

# ページ設定
st.set_page_config(page_title="FX リアルタイム予測", layout="wide")
st_autorefresh(interval=60 * 1000, key="fxtracker")

st.title("💹 ドル円 リアルタイム・トレンド予測")

# 改良版：リトライ機能付きデータ取得
@st.cache_data(ttl=60)
def get_latest_data():
    ticker = "USDJPY=X"
    # 3回までリトライを試みる
    for attempt in range(3):
        try:
            df = yf.download(ticker, period="1d", interval="1m", progress=False)
            if not df.empty:
                # 10分足へリサンプリング
                df_10m = df['Close'].resample('10min').last().dropna().reset_index()
                df_10m.columns = ['ds', 'y']
                df_10m['ds'] = df_10m['ds'].dt.tz_localize(None)
                return df_10m
        except Exception as e:
            time.sleep(1) # 失敗したら1秒待機
    return None

try:
    df = get_latest_data()

    if df is None or len(df) < 2:
        st.error("データ取得エラー: 現在データが取得できません。市場環境か通信の一時的な不具合です。しばらく待つと復旧することが多いです。")
        st.write("もしエラーが続く場合、yfinanceの接続制限を受けている可能性があります。")
    else:
        # 最新価格の表示
        current_price = df['y'].iloc[-1]
        last_update = df['ds'].iloc[-1].strftime('%H:%M:%S')
        st.subheader(f"現在価格: {current_price:.3f} JPY (更新: {last_update})")

        # 予測
        model = Prophet(changepoint_prior_scale=0.01, daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
        model.fit(df)
        future = model.make_future_dataframe(periods=3, freq='10min')
        forecast = model.predict(future)

        # グラフ
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="実績", line=dict(color='#00CF80', width=2)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="予測", line=dict(dash='dash', color='#FF4B4B')))
        fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

        st.info(f"次回の10分足 予測ターゲット: {forecast['yhat'].iloc[-1]:.3f} JPY")

except Exception as e:
    st.error(f"予期せぬエラー: {e}")
