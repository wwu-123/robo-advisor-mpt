import os
import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

# 設定 Matplotlib 風格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 網頁頁面標題與寬度設定
st.set_page_config(page_title="理財機器人 - 動態資產配置系統", layout="wide")

st.title("🤖 理財機器人：動態問卷與資產配置最佳化系統")
st.markdown("本系統整合**投資人風險屬性動態評估 (Investor Profiling)** 與 **現代投資組合理論 (MPT)**，即時計算最佳資產配置[cite: 1, 2]。")

# ==========================================
# 1. 自動取得相對路徑與資料載入模組
# ==========================================
# 取得目前 app.py 檔案所在的資料夾相對目錄
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXCEL_PATH = os.path.join(BASE_DIR, '20261005095516.xlsx')

st.sidebar.header("📁 1. 金融資料設定")
uploaded_file = st.sidebar.file_uploader("可自訂上傳 Excel/CSV 價格資料檔", type=["xlsx", "csv"])

@st.cache_data
def load_data(file):
    """
    資料載入函數：若使用者有上傳檔案則讀取上傳檔，否則自動讀取專案同目錄下的預設 Excel 檔
    """
    if file is not None:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    else:
        # 使用相對路徑讀取專案資料夾內的預設 Excel 檔
        if os.path.exists(DEFAULT_EXCEL_PATH):
            return pd.read_excel(DEFAULT_EXCEL_PATH)
        else:
            st.error(f"找不到預設資料檔 `20261005095516.xlsx`，請確認該檔案是否已上傳至專案根目錄中[cite: 5, 7]。")
            return None

df_raw = load_data(uploaded_file)

if df_raw is None:
    st.stop()

# 顯示資料載入狀態提醒
if uploaded_file is None:
    st.sidebar.success("已自動載入專案預設資料檔：`20261005095516.xlsx`[cite: 2]")
else:
    st.sidebar.success(f"已成功載入自訂檔案：`{uploaded_file.name}`")

# 資料轉置與報酬率計算
df_pivot = df_raw.pivot(index='年月日', columns='證券代碼', values='收盤價(元)')
df_pivot.index = pd.to_datetime(df_pivot.index.astype(str))
returns = df_pivot.pct_change().dropna()

mean_returns_monthly = returns.mean()
cov_matrix_monthly = returns.cov()

# 投資組合績效計算函數 (年化因子 = 12)
def portfolio_performance(weights):
    ret = np.sum(mean_returns_monthly * weights) * 12
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_monthly * 12, weights)))
    return ret, vol

# ==========================================
# 2. 主頁面：動態問卷模組
# ==========================================
st.subheader("📋 2. 投資人風險屬性動態評估問卷")

col1, col2, col3 = st.columns(3)

with col1:
    q1 = st.radio("1. 您的投資年齡層？", 
                  ["60歲以上 (1分)", "40-59歲 (2分)", "40歲以下 (3分)"])
    score1 = 1 if "1分" in q1 else (2 if "2分" in q1 else 3)

with col2:
    q2 = st.radio("2. 您的投資主要目標？", 
                  ["資產保值與穩定收益 (1分)", "穩健成長 (2分)", "追求最大資本利得 (3分)"])
    score2 = 1 if "1分" in q2 else (2 if "2分" in q2 else 3)

with col3:
    q3 = st.radio("3. 若短期出現 20% 虧損，您會？", 
                  ["全部賣出止損 (1分)", "持續持有觀察 (2分)", "逢低加碼買進 (3分)"])
    score3 = 1 if "1分" in q3 else (2 if "2分" in q3 else 3)

total_score = score1 + score2 + score3

# 風險等級判定與 MPT 目標對接
if total_score <= 4:
    profile_name = "保守型 (Conservative)"
    target_type = "MVP (最小變異數)"
    gamma = 10.0
elif total_score <= 7:
    profile_name = "穩健型 (Moderate)"
    target_type = "Max Sharpe (最大夏普比率)"
    gamma = 3.0
else:
    profile_name = "積極型 (Aggressive)"
    target_type = "Max Return (最大報酬)"
    gamma = 0.5

st.info(f"💡 **問卷總分：{total_score} 分** ｜ 屬性判定：**{profile_name}** ｜ MPT 最佳化策略：**{target_type}**")

# ==========================================
# 3. MPT 最佳化演算法計算
# ==========================================
num_assets = len(mean_returns_monthly)
bounds = tuple((0, 1) for _ in range(num_assets))
init_guess = num_assets * [1.0 / num_assets]

if "MVP" in target_type:
    obj_func = lambda w: portfolio_performance(w)[1]
elif "Max Sharpe" in target_type:
    obj_func = lambda w: -(portfolio_performance(w)[0] - 0.015) / portfolio_performance(w)[1]
else:
    obj_func = lambda w: -(portfolio_performance(w)[0] - 0.5 * gamma * (portfolio_performance(w)[1]**2))

res = sco.minimize(obj_func, init_guess, method='SLSQP', bounds=bounds, 
                   constraints=({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}))

w_opt = res.x
ret_opt, vol_opt = portfolio_performance(w_opt)
sharpe_opt = (ret_opt - 0.015) / vol_opt

# ==========================================
# 4. 圖表視覺化與結果呈現
# ==========================================
st.subheader("📊 3. 動態最佳化資產配置報告")

m1, m2, m3 = st.columns(3)
m1.metric("預期年化報酬率", f"{ret_opt*100:.2f}%")
m2.metric("預期年化波動度 (風險)", f"{vol_opt*100:.2f}%")
m3.metric("夏普比率 (Sharpe)", f"{sharpe_opt:.2f}")

chart_col1, chart_col2 = st.columns(2)

# (A) 資產配置圓餅圖
with chart_col1:
    st.write("#### 🎯 最適資產配置比例")
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    df_weights = pd.Series(w_opt, index=df_pivot.columns)
    df_weights_filter = df_weights[df_weights > 0.001]  # 濾除 0% 的權重
    ax1.pie(df_weights_filter, labels=df_weights_filter.index, autopct='%1.1f%%', startangle=140)
    ax1.axis('equal')
    st.pyplot(fig1)

# (B) 效率前緣圖 (Efficient Frontier)
with chart_col2:
    st.write("#### 📈 效率前緣與當前點位")
    asset_rets = mean_returns_monthly * 12
    target_returns = np.linspace(asset_rets.min(), asset_rets.max(), 50)
    ef_vols = []
    
    for tr in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: portfolio_performance(x)[0] - tr}
        )
        r_ef = sco.minimize(lambda x: portfolio_performance(x)[1], init_guess, 
                            method='SLSQP', bounds=bounds, constraints=constraints)
        ef_vols.append(r_ef.fun if r_ef.success else np.nan)

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(ef_vols, target_returns, 'b--', label='Efficient Frontier')
    ax2.scatter(vol_opt, ret_opt, color='red', s=150, zorder=5, label=f'Your Portfolio ({profile_name.split()[0]})')
    ax2.set_xlabel('Portfolio Risk / Volatility (σ)')
    ax2.set_ylabel('Expected Return (E[R])')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig2)

# 明細表格
st.write("#### 📋 各標的配置比重明細")
df_display = pd.DataFrame({
    '證券代碼與標的名稱': df_pivot.columns,
    '權重比例 (%)': [f"{w*100:.2f}%" for w in w_opt]
})
st.dataframe(df_display, use_container_width=True)