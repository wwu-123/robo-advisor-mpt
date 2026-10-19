import os
import streamlit as st
import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt

# 設定 Matplotlib 風格
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# 網頁頁面標題與寬度設定
st.set_page_config(page_title="理財機器人 - 多元動態資產配置系統", layout="wide")

st.title("🤖 理財機器人：多元風險問卷與資產配置最佳化系統")
st.markdown("本系統整合**多元投資人風險屬性動態評估 (Investor Profiling)** 與 **現代投資組合理論 (MPT)**，即時計算專屬最佳資產配置[cite: 1, 2]。")

# ==========================================
# 1. 自動取得相對路徑與資料載入模組
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXCEL_PATH = os.path.join(BASE_DIR, '20261005095516.xlsx')

st.sidebar.header("📁 1. 金融資料設定")
uploaded_file = st.sidebar.file_uploader("可自訂上傳 Excel/CSV 價格資料檔", type=["xlsx", "csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    else:
        if os.path.exists(DEFAULT_EXCEL_PATH):
            return pd.read_excel(DEFAULT_EXCEL_PATH)
        else:
            st.error("找不到預設資料檔 `20261005095516.xlsx`，請確認檔案是否在根目錄中[cite: 5, 7]。")
            return None

df_raw = load_data(uploaded_file)

if df_raw is None:
    st.stop()

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

def portfolio_performance(weights):
    ret = np.sum(mean_returns_monthly * weights) * 12
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_monthly * 12, weights)))
    return ret, vol

# ==========================================
# 2. 多元動態評估問卷 (不顯示選項分數)
# ==========================================
st.subheader("📋 2. 投資人風險屬性動態問卷")
st.caption("請根據您的真實投資狀況選擇最符合的選項，系統將自動計算您的風險承受類型：")

col1, col2 = st.columns(2)

with col1:
    q1 = st.radio(
        "1. 您的年齡與人生階段狀況？",
        ["65歲以上 / 已退休或準備退休", "50–64歲 / 步入職涯成熟期", "35–49歲 / 家庭與事業穩定成長期", "34歲以下 / 職場新秀或個人衝刺期"]
    )
    
    q2 = st.radio(
        "2. 您進行這筆投資的主要目的為何？",
        ["抵抗通膨，維持資產購買力與安全", "賺取穩定定期配息，補充日常現金流", "穩健累積資產，實現中長期財務目標", "追求資產大幅增值與高資本利得"]
    )
    
    q3 = st.radio(
        "3. 您過去的投資經驗與工具熟悉度？",
        ["幾乎沒有投資經驗，僅習慣銀行定存/儲蓄險", "主要投資定存、債券或保守型基金", "具備股票、ETF、共同基金等多元投資經驗", "熟練操作股票、衍生性商品、高波動標的"]
    )

with col2:
    q4 = st.radio(
        "4. 若未來一年市場大跌，資產出現 20% 短期帳面虧損，您的反應是？",
        ["極度焦慮，會立即全部賣出以防虧損擴大", "感到不安，會減碼賣出大部分高風險資產", "保持觀望，繼續持有並等待市場回升", "視為難得的加碼時機，逢低分批加碼"]
    )
    
    q5 = st.radio(
        "5. 您對投資組合報酬與波動度的偏好？",
        ["偏好極低波動，即使年化報酬率較低也能接受", "接受微幅波動，換取高於定存的合理收益", "追求報酬與風險的平衡，期望長期穩定增長", "願意承擔較大價格波動，以換取極致的總報酬"]
    )

# 內部分數映射邏輯 (介面隱藏)
score_map_1 = {"65歲以上 / 已退休或準備退休": 1, "50–64歲 / 步入職涯成熟期": 2, "35–49歲 / 家庭與事業穩定成長期": 3, "34歲以下 / 職場新秀或個人衝刺期": 4}
score_map_2 = {"抵抗通膨，維持資產購買力與安全": 1, "賺取穩定定期配息，補充日常現金流": 2, "穩健累積資產，實現中長期財務目標": 3, "追求資產大幅增值與高資本利得": 4}
score_map_3 = {"幾乎沒有投資經驗，僅習慣銀行定存/儲蓄險": 1, "主要投資定存、債券或保守型基金": 2, "具備股票、ETF、共同基金等多元投資經驗": 3, "熟練操作股票、衍生性商品、高波動標的": 4}
score_map_4 = {"極度焦慮，會立即全部賣出以防虧損擴大": 1, "感到不安，會減碼賣出大部分高風險資產": 2, "保持觀望，繼續持有並等待市場回升": 3, "視為難得的加碼時機，逢低分批加碼": 4}
score_map_5 = {"偏好極低波動，即使年化報酬率較低也能接受": 1, "接受微幅波動，換取高於定存的合理收益": 2, "追求報酬與風險的平衡，期望長期穩定增長": 3, "願意承擔較大價格波動，以換取極致的總報酬": 4}

total_score = score_map_1[q1] + score_map_2[q2] + score_map_3[q3] + score_map_4[q4] + score_map_5[q5]

# 擴充為 5 種投資人風格 (Profiles)
if total_score <= 7:
    profile_name = "🛡️ 極度保守型 (Very Conservative)"
    target_type = "MVP (最小變異數)"
    gamma = 15.0
elif total_score <= 11:
    profile_name = "🌿 保守型 (Conservative)"
    target_type = "Low Volatility (低波動偏好)"
    gamma = 6.0
elif total_score <= 14:
    profile_name = "⚖️ 穩健型 (Moderate)"
    target_type = "Max Sharpe (最大夏普比率)"
    gamma = 2.5
elif total_score <= 17:
    profile_name = "🚀 成長型 (Growth)"
    target_type = "High Growth (資本成長)"
    gamma = 1.0
else:
    profile_name = "🔥 積極型 (Aggressive)"
    target_type = "Max Return (最大報酬)"
    gamma = 0.2

st.info(f"💡 評估結果：屬性判定為 **{profile_name}** ｜ MPT 最佳化策略：**{target_type}**")

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
    # 效用函數: U = Return - 0.5 * gamma * Risk^2
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
    df_weights_filter = df_weights[df_weights > 0.001]
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
    ax2.scatter(vol_opt, ret_opt, color='red', s=150, zorder=5, label=f'Your Portfolio')
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