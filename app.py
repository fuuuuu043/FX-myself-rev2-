import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta

st.set_page_config(page_title="FX リアルタイム予測", layout="wide")
# 1分ごとに画面を更新
st_autorefresh(interval=60 * 1000, key="fxtracker")

st.title("💹 ドル円 最新トレンド予測")
st.caption(f"現在のシステム時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

@st.cache_data(ttl=60)
def get_latest_data():
    try:
        # 1分足が不安定なため、より安定している「2分足」または「5分足」で取得を試みます
        # periodを「1d(今日1日)」に絞ることで、最新データの取得精度を上げます
        df = yf.download("USDJPY=X", period="1d", interval="2m")
        
        if df.empty or len(df) < 5:
            # 1dで取れない場合、保険として過去2日分を取得
            df = yf.download("USDJPY=X", period="2d", interval="5m")

        if df.empty:
            return None
            
        # Prophet用にカラムを整形
        df_resampled = df['Close'].resample('10min').last().dropna().reset_index()
        df_resampled.columns = ['ds', 'y']
        df_resampled['ds'] = df_resampled['ds'].dt.tz_localize(None)
        return df_resampled
    except Exception as e:
        st.error(f"データ取得エラー: {e}")
        return None

try:
    df = get_latest_data()

    if df is None:
        st.warning("現在、最新データを取得待ちです。市場の動きがないか、APIの制限がかかっている可能性があります。")
    else:
        # 最新のデータ時刻を表示（ここで21:30に近いか確認できます）
        last_dt = df['ds'].iloc[-1]
        st.info(f"データ参照時刻: {last_dt.strftime('%Y-%m-%d %H:%M:%S')}")

        # 予測モデル
        model = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True)
        model.fit(df)
        
        future = model.make_future_dataframe(periods=3, freq='10min')
        forecast = model.predict(future)
        
        # グラフ表示
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="実績", line=dict(color='#00CF80', width=3)))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="予測", line=dict(dash='dash', color='#FF4B4B')))
        
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # 予測結果の強調表示
        latest_p = df['y'].iloc[-1]
        target_p = forecast['yhat'].iloc[-1]
        st.metric("現在価格", f"{latest_p:.3f} JPY")
        st.write(f"👉 10分後の予測ターゲット: **{target_p:.3f} JPY**")

except Exception as e:
    st.error(f"解析エラーが発生しました: {e}")
