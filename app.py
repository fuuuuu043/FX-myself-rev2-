import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

# ページのタイトル設定
st.set_page_config(page_title="FX 10分足予測", layout="wide")
st.title("💹 ドル円 10分足トレンド予測 (基礎版)")

# 1. データの取得（GASのUrlFetchAppのような役割）
@st.cache_data(ttl=600)  # 10分間データをキャッシュ（無料枠を節約）
def get_data():
    # 過去5日分の1分足データを取得
    df = yf.download("USDJPY=X", period="5d", interval="1m")
    # 10分足に集計し直す
    df_10m = df['Close'].resample('10min').last().dropna().reset_index()
    df_10m.columns = ['ds', 'y']
    df_10m['ds'] = df_10m['ds'].dt.tz_localize(None) # 時区設定の解除
    return df_10m

try:
    df = get_data()
    
    # 2. 予測モデルの作成（学習・計算）
    # 学習機能を使わなくても、Prophetが統計的にトレンドを算出します
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df)
    
    # 未来1時間分（10分×6回）の枠を作成して予測
    future = model.make_future_dataframe(periods=6, freq='10min')
    forecast = model.predict(future)
    
    # 3. グラフの表示
    fig = go.Figure()
    # 実測値（青）
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="現在までの価格"))
    # 予測値（オレンジの点線）
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="予測値", line=dict(dash='dash')))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 4. 数値の表示
    latest_price = df['y'].iloc[-1]
    predicted_price = forecast['yhat'].iloc[-1]
    diff = predicted_price - latest_price
    
    col1, col2 = st.columns(2)
    col1.metric("現在の価格", f"{latest_price:.3f} JPY")
    col2.metric("10分後の予想", f"{predicted_price:.3f} JPY", f"{diff:.3f}")

except Exception as e:
    st.error(f"エラーが発生しました。時間を置いて再度お試しください: {e}")
