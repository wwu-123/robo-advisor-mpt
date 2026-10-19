import pandas as pd
import numpy as np
import scipy.optimize as sco

def run_dynamic_questionnaire():
    """
    動態風險問卷系統：根據使用者輸入分數，即時計算專屬最佳化參數
    """
    print("\n==================================================")
    print("      理財機器人：投資人風險屬性動態評估問卷      ")
    print("==================================================")
    
    questions = [
        {
            "q": "1. 您的投資年齡層為何？",
            "options": {"1": "60歲以上 (1分)", "2": "40-59歲 (2分)", "3": "40歲以下 (3分)"},
            "scores": {"1": 1, "2": 2, "3": 3}
        },
        {
            "q": "2. 您的投資主要目標為何？",
            "options": {"1": "資產保值與穩定收益 (1分)", "2": "穩健成長與部分收益 (2分)", "3": "追求最大資本利得 (3分)"},
            "scores": {"1": 1, "2": 2, "3": 3}
        },
        {
            "q": "3. 當市場波動導致投資組合短期虧損 20% 時，您的反應是？",
            "options": {"1": "全部賣出止損 (1分)", "2": "持續持有並觀察 (2分)", "3": "逢低加碼買進 (3分)"},
            "scores": {"1": 1, "2": 2, "3": 3}
        }
    ]
    
    total_score = 0
    for idx, item in enumerate(questions, 1):
        print(f"\n{item['q']}")
        for opt_key, opt_val in item['options'].items():
            print(f"  [{opt_key}] {opt_val}")
        
        while True:
            choice = input("請選擇 (1/2/3): ").strip()
            if choice in item['scores']:
                total_score += item['scores'][choice]
                break
            else:
                print("輸入無效，請輸入 1, 2 或 3。")
                
    print("\n--------------------------------------------------")
    print(f"評估完成！您的總得分為：{total_score} 分")
    
    # 根據分數動態判定風險類型與目標函數參數
    if total_score <= 4:
        profile_name = "保守型 (Conservative)"
        target_type = "MVP"  # 最小變異數組合
        gamma = 10.0
    elif total_score <= 7:
        profile_name = "穩健型 (Moderate)"
        target_type = "Max_Sharpe"  # 最大夏普比率組合
        gamma = 3.0
    else:
        profile_name = "積極型 (Aggressive)"
        target_type = "Max_Return"  # 最大報酬率/低風險規避
        gamma = 0.5
        
    print(f"判定風險等級：【{profile_name}】")
    print(f"帶入 MPT 最佳化模型參數：Target = {target_type}, Gamma = {gamma}")
    print("--------------------------------------------------\n")
    
    return profile_name, target_type, gamma


# ==================================================
# 帶入 20261005095516.xlsx 資料進行動態最佳化運算
# ==================================================
def calculate_optimal_portfolio(file_path, target_type, gamma):
    df_raw = pd.read_excel(file_path)
    df_pivot = df_raw.pivot(index='年月日', columns='證券代碼', values='收盤價(元)')
    df_pivot.index = pd.to_datetime(df_pivot.index.astype(str))
    returns = df_pivot.pct_change().dropna()
    
    mean_rets = returns.mean()
    cov_mat = returns.cov()
    num_assets = len(mean_rets)
    bounds = tuple((0, 1) for _ in range(num_assets))
    init_guess = num_assets * [1.0 / num_assets]
    
    def portfolio_perf(w):
        r = np.sum(mean_rets * w) * 12
        v = np.sqrt(np.dot(w.T, np.dot(cov_mat * 12, w)))
        return r, v
    
    # 依問卷結果動態切換最佳化目標函數
    if target_type == "MVP":
        # 極小化風險
        obj_func = lambda w: portfolio_perf(w)[1]
    elif target_type == "Max_Sharpe":
        # 極大化夏普比率 (rf=1.5%)
        obj_func = lambda w: -(portfolio_perf(w)[0] - 0.015) / portfolio_perf(w)[1]
    else:
        # 效用極大化 Utility = Return - 0.5 * Gamma * Risk^2
        obj_func = lambda w: -(portfolio_perf(w)[0] - 0.5 * gamma * (portfolio_perf(w)[1]**2))
        
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    res = sco.minimize(obj_func, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    
    ret, vol = portfolio_perf(res.x)
    return res.x, ret, vol, df_pivot.columns

# 主程序執行流程範例
if __name__ == '__main__':
    excel_path = r"C:\Users\ASUS\OneDrive\桌面\分析\20261005095516.xlsx"
    
    # 1. 執行問卷收集動態輸入
    profile_name, target_type, gamma = run_dynamic_questionnaire()
    
    # 2. 根據問卷動態計算權重
    weights, ret, vol, asset_names = calculate_optimal_portfolio(excel_path, target_type, gamma)
    
    # 3. 輸出該問卷對應的專屬資產配置結果
    print(f"=== 【{profile_name}】動態生成之最佳資產配置 ===")
    print(f"預期年化報酬率: {ret*100:.2f}%")
    print(f"預期年化波動度: {vol*100:.2f}%")
    print("資產配置權重比例:")
    for asset, w in zip(asset_names, weights):
        print(f"  * {asset}: {w*100:.2f}%")