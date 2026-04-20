import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="FX 10分足予測", layout="wide")
st.title("💹 ドル円 10分足トレンド予測 (修正版)")

@st.cache_data(ttl=600)
def get_data():
    # 取得期間を「5d（5日間）」から「7d（7日間）」に少し広げます
    df = yf.download("USDJPY=X", period="7d", interval="1m")
    
    # データの存在チェック
    if df.empty:
        return pd.DataFrame()

    # 10分足に集計
    df_10m = df['Close'].resample('10min').last().dropna().reset_index()
    df_10m.columns = ['ds', 'y']
    df_10m['ds'] = df_10m['ds'].dt.tz_localize(None)
    return df_10m

try:
    df = get_data()

    # データが予測に必要な最低ライン（ここでは念のため20行以上）あるかチェック
    if len(df) < 20:
        st.warning(f"現在データ取得中です。十分なデータが集まるまでお待ちください（現在のデータ数: {len(df)}）")
        st.info("※週末や市場が閉まっている時間帯は新しいデータが取得できない場合があります。")
    else:
        # 予測モデルの作成
        model = Prophet(
            daily_seasonality=True, 
            weekly_seasonality=True, 
            yearly_seasonality=False,
            changepoint_prior_scale=0.05 # 柔軟性を少し調整
        )
        model.fit(df)
        
        # 未来1時間分を予測
        future = model.make_future_dataframe(periods=6, freq='10min')
        forecast = model.predict(future)
        
        # グラフ表示
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], name="実績価格", line=dict(color='deepskyblue')))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="予測トレンド", line=dict(dash='dash', color='orange')))
        
        fig.update_layout(xaxis_rangeslider_visible=False, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # 指標の表示
        latest_price = df['y'].iloc[-1]
        predicted_price = forecast['yhat'].iloc[-1]
        diff = predicted_price - latest_price
        
        col1, col2, col3 = st.columns(3)
        col1.metric("現在の価格", f"{latest_price:.3f}")
        col2.metric("10分後の予想", f"{predicted_price:.3f}", f"{diff:.3f}")
        col3.write(f"最終更新: {df['ds'].iloc[-1]}")

except Exception as e:
    st.error(f"システムエラー: {e}")
