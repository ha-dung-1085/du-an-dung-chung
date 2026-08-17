import os
import datetime
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ==============================================================================
# 🛠️ KHU VỰC CẤU HÌNH DÀNH CHO NGƯỜI DÙNG
# ==============================================================================

FILE_PATH = r'C:\Users\pc-008\Desktop\du-an-dung-chung\Ket_Qua_Phan_Tich_SO SÁNH PP - ĐTCS\Chạy lại\0. KẾT QUẢ PHÂN TÍCH MẪU_ĐTCS 2.xlsx'
SHEET_NAME = 'FINAL (3)'

TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = "Ket_Qua_Phan_Tich_So_Sanh"
OUTPUT_DIR = f"{BASE_OUTPUT_DIR}_{TIMESTAMP}"

DECIMAL_PLACES = 2  

ANALYTE_PAIRS = {
    'Chỉ số H': ('H_Index_PPA', 'H_Index_PPB'),
    'Chỉ số L': ('L_Index_PPA', 'L_Index_PPB'),
    'Chỉ số I': ('I_Index_PPA', 'I_Index_PPB'),
    'Potassium (K+)': ('K_PPA', 'K_PPB'),
    'Sodium (Na+)': ('Na_PPA', 'Na_PPB'),
    'Chloride (Cl-)': ('Cl_PPA', 'Cl_PPB'),
    'AST': ('AST_PPA', 'AST_PPB'),
    'LDH': ('LDH_PPA', 'LDH_PPB'),
    'TEST': ('TEST_PPA', 'TEST_PPB')
}

INTEGER_ANALYTES = ['Chỉ số H', 'Chỉ số L', 'Chỉ số I']

BIAS_LIMITS = {
    'AST': {
        'Pct': {'Optimal': 2.7, 'Desirable': 5.4, 'Minimum': 8.0},
        'Abs': {'Optimal': 1.5, 'Desirable': 3.0, 'Minimum': 4.5}
    },
    'Potassium (K+)': {
        'Pct': {'Optimal': 0.8, 'Desirable': 1.7, 'Minimum': 2.5},
        'Abs': {'Optimal': 0.05, 'Desirable': 0.10, 'Minimum': 0.15}
    },
    'LDH': {
        'Pct': {'Optimal': 1.6, 'Desirable': 3.1, 'Minimum': 4.7},
        'Abs': {'Optimal': None, 'Desirable': None, 'Minimum': None}
    }
}

PPA_PREDICT_VALUES = {
    'AST': [32.0, 40.0, 400.0],
    'Potassium (K+)': [3.0, 3.4, 4.5, 6.0],
    'LDH': [135.0, 225.0]
}

REQUIRED_6_ANALYTES = ['Chỉ số H', 'Chỉ số I', 'Chỉ số L', 'AST', 'Potassium (K+)', 'LDH']

# ==============================================================================
# HÀM BỔ TRỢ ĐỊNH DẠNG SỐ VÀ THỐNG KÊ
# ==============================================================================
def fmt(val, decimals=DECIMAL_PLACES):
    if val is None or pd.isna(val):
        return "N/A"
    if isinstance(val, (int, float, np.number)):
        return f"{val:.{decimals}f}"
    return str(val)

def check_normality(data):
    if len(data) < 3:
        return False, 0.0
    stat, p_val = stats.shapiro(data)
    return (p_val >= 0.05), p_val

def generalized_esd_test(data, alpha=0.05, max_outliers=None):
    n = len(data)
    if max_outliers is None:
        max_outliers = max(1, int(0.2 * n))
    
    data_list = list(data)
    indices = list(range(n))
    outlier_indices = []
    
    for i in range(1, max_outliers + 1):
        if len(data_list) < 3:
            break
        mean_val = np.mean(data_list)
        std_val = np.std(data_list, ddof=1)
        if std_val == 0:
            break
        
        abs_diff = [abs(x - mean_val) for x in data_list]
        max_idx = np.argmax(abs_diff)
        R = abs_diff[max_idx] / std_val
        
        curr_n = len(data_list)
        p = 1 - alpha / (2 * curr_n)
        t_crit = stats.t.ppf(p, curr_n - 2)
        lambda_crit = ((curr_n - 1) * t_crit) / np.sqrt((curr_n - 2 + t_crit**2) * curr_n)
        
        if R > lambda_crit:
            outlier_indices.append(indices[max_idx])
            data_list.pop(max_idx)
            indices.pop(max_idx)
        else:
            break
            
    return outlier_indices

def get_analyte_variants(ppa_v, ppb_v, diff_v, df_analyte_valid, is_integer):
    if is_integer:
        return [{
            'suffix': '',
            'ppa': ppa_v,
            'ppb': ppb_v,
            'diff': diff_v,
            'df_sub': df_analyte_valid,
            'is_normal': False,
            'p_shapiro': 'N/A',
            'norm_desc': 'Biến số nguyên (Không kiểm định số lạc/phân bố chuẩn)',
            'n_detected': 0,
            'n_removed': 0,
            'outlier_idx': []
        }]

    is_norm_init, p_init = check_normality(diff_v)
    outlier_idx = generalized_esd_test(diff_v)
    n_detected = len(outlier_idx)

    if n_detected == 0:
        norm_desc = "Chuẩn ban đầu (Không có số lạc)" if is_norm_init else "Không chuẩn (Không có số lạc)"
        return [{
            'suffix': '',
            'ppa': ppa_v,
            'ppb': ppb_v,
            'diff': diff_v,
            'df_sub': df_analyte_valid,
            'is_normal': is_norm_init,
            'p_shapiro': p_init,
            'norm_desc': norm_desc,
            'n_detected': 0,
            'n_removed': 0,
            'outlier_idx': []
        }]

    variants = []

    # 1. Tập Dữ Liệu Gốc
    norm_desc_orig = f"Dữ liệu gốc (Phát hiện {n_detected} số lạc, giữ nguyên)"
    variants.append({
        'suffix': ' [Gốc]',
        'ppa': ppa_v,
        'ppb': ppb_v,
        'diff': diff_v,
        'df_sub': df_analyte_valid,
        'is_normal': is_norm_init,
        'p_shapiro': p_init,
        'norm_desc': norm_desc_orig,
        'n_detected': n_detected,
        'n_removed': 0,
        'outlier_idx': outlier_idx
    })

    # 2. Tập Dữ Liệu Đã Loại Số Lạc
    clean_diff = np.delete(diff_v, outlier_idx)
    clean_ppa = np.delete(ppa_v, outlier_idx)
    clean_ppb = np.delete(ppb_v, outlier_idx)
    clean_df = df_analyte_valid.drop(df_analyte_valid.index[outlier_idx]).reset_index(drop=True)

    is_norm_clean, p_clean = check_normality(clean_diff)
    norm_desc_clean = f"Đã loại {n_detected} số lạc GESD (" + ("Chuẩn" if is_norm_clean else "Không chuẩn") + ")"
    
    variants.append({
        'suffix': f' [Loại {n_detected} số lạc]',
        'ppa': clean_ppa,
        'ppb': clean_ppb,
        'diff': clean_diff,
        'df_sub': clean_df,
        'is_normal': is_norm_clean,
        'p_shapiro': p_clean,
        'norm_desc': norm_desc_clean,
        'n_detected': n_detected,
        'n_removed': n_detected,
        'outlier_idx': outlier_idx
    })

    return variants

def perform_difference_test(ppa_data, ppb_data, is_normal):
    if is_normal:
        test_name = "Paired t-test"
        stat, p_val = stats.ttest_rel(ppa_data, ppb_data)
    else:
        test_name = "Wilcoxon Signed-Rank test"
        diff = ppb_data - ppa_data
        if np.all(diff == 0):
            return test_name, 0.0, 1.0, "Không có sự khác biệt (các cặp giá trị giống hệt nhau)"
        stat, p_val = stats.wilcoxon(ppa_data, ppb_data)
    
    sig_text = "Có sự khác biệt có ý nghĩa thống kê (p < 0.05)" if p_val < 0.05 else "Khác biệt không có ý nghĩa thống kê (p >= 0.05)"
    return test_name, stat, p_val, sig_text

def classify_clsi_bias(bias_val, ci_low, ci_high, b_max):
    if pd.isna(bias_val) or pd.isna(ci_low) or pd.isna(ci_high) or pd.isna(b_max) or b_max is None:
        return "N/A", "Không áp dụng / Không khai báo B_max"
    
    abs_bias = abs(bias_val)
    abs_ci_max = max(abs(ci_low), abs(ci_high))
    abs_ci_min = min(abs(ci_low), abs(ci_high))
    contains_zero = (ci_low <= 0 <= ci_high) or (ci_high <= 0 <= ci_low)
    
    if abs_ci_max <= b_max:
        if contains_zero:
            return 'A', 'Hoàn toàn chấp nhận được (Bias không khác 0 và CI <= B_max)'
        else:
            return 'B', 'Chấp nhận được (Bias có ý nghĩa nhưng CI <= B_max)'
    elif abs_bias <= b_max < abs_ci_max:
        return 'C', 'Nghi ngờ chấp nhận (Bias <= B_max nhưng CI vượt B_max)'
    elif abs_ci_min <= b_max < abs_bias:
        return 'D', 'Nghi ngờ không đạt (Bias > B_max nhưng CI vẫn chạm B_max)'
    else:
        return 'E', 'Không chấp nhận được (CI vượt hoàn toàn B_max)'

def passing_bablok_fit_and_predict(x, y, targets_x=None, n_boot=1000):
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] != x[j]:
                s = (y[j] - y[i]) / (x[j] - x[i])
                if s != -1:
                    slopes.append(s)
    
    slope = np.median(slopes) if len(slopes) > 0 else 1.0
    intercept = np.median(y - slope * x)
    
    target_list = []
    if targets_x is not None:
        if isinstance(targets_x, (int, float, np.number)):
            target_list = [float(targets_x)]
        elif isinstance(targets_x, (list, tuple, np.ndarray)):
            target_list = [float(val) for val in targets_x]

    boot_slopes, boot_intercepts = [], []
    boot_preds = {tx: {'y_preds': [], 'bias_abs': [], 'bias_pct': []} for tx in target_list}

    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        xb, yb = x[idx], y[idx]
        sub_slopes = []
        for i in range(n):
            for j in range(i + 1, n):
                if xb[i] != xb[j]:
                    s = (yb[j] - yb[i]) / (xb[j] - xb[i])
                    if s != -1:
                        sub_slopes.append(s)
        if len(sub_slopes) > 0:
            bs = np.median(sub_slopes)
            bi = np.median(yb - bs * xb)
            boot_slopes.append(bs)
            boot_intercepts.append(bi)

            for tx in target_list:
                y_p = bi + bs * tx
                b_abs = y_p - tx
                b_pct = (b_abs / tx) * 100.0 if tx != 0 else 0.0
                boot_preds[tx]['y_preds'].append(y_p)
                boot_preds[tx]['bias_abs'].append(b_abs)
                boot_preds[tx]['bias_pct'].append(b_pct)
            
    slope_ci = np.percentile(boot_slopes, [2.5, 97.5]) if boot_slopes else (slope, slope)
    intercept_ci = np.percentile(boot_intercepts, [2.5, 97.5]) if boot_intercepts else (intercept, intercept)

    predict_results_list = []
    for tx in target_list:
        p_data = boot_preds[tx]
        if len(p_data['y_preds']) > 0:
            predict_results_list.append({
                'target_x': tx,
                'y_pred': np.median(p_data['y_preds']),
                'y_pred_ci': np.percentile(p_data['y_preds'], [2.5, 97.5]),
                'bias_abs': np.median(p_data['bias_abs']),
                'bias_abs_ci': np.percentile(p_data['bias_abs'], [2.5, 97.5]),
                'bias_pct': np.median(p_data['bias_pct']),
                'bias_pct_ci': np.percentile(p_data['bias_pct'], [2.5, 97.5])
            })
    
    return slope, intercept, slope_ci, intercept_ci, predict_results_list

def plot_bland_altman_with_histogram(x_vals, diff, bias, bias_ci, loa_low, loa_low_ci, loa_high, loa_high_ci, title, xlabel, ylabel, unit_str, filepath):
    fig = plt.figure(figsize=(9.5, 6), dpi=300)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3.5, 1], wspace=0.08)

    ax_scatter = plt.subplot(gs[0])
    ax_hist = plt.subplot(gs[1], sharey=ax_scatter)

    ax_scatter.scatter(x_vals, diff, color='#2b5c8f', alpha=0.7, edgecolors='k', label='Giá trị khảo sát')
    
    l_bias = ax_scatter.axhline(bias, color='red', linestyle='-', linewidth=1.5, 
                                label=f'Bias: {fmt(bias)}{unit_str} [95%CI: {fmt(bias_ci[0])}, {fmt(bias_ci[1])}]')
    
    l_loa_h = ax_scatter.axhline(loa_high, color='green', linestyle='--', linewidth=1.5, 
                                 label=f'Upper LoA: {fmt(loa_high)}{unit_str} [95%CI: {fmt(loa_high_ci[0])}, {fmt(loa_high_ci[1])}]')
    
    l_loa_l = ax_scatter.axhline(loa_low, color='green', linestyle='--', linewidth=1.5, 
                                 label=f'Lower LoA: {fmt(loa_low)}{unit_str} [95%CI: {fmt(loa_low_ci[0])}, {fmt(loa_low_ci[1])}]')

    ax_scatter.axhline(bias_ci[0], color='red', linestyle=':', alpha=0.6, linewidth=1)
    ax_scatter.axhline(bias_ci[1], color='red', linestyle=':', alpha=0.6, linewidth=1)
    
    ax_scatter.axhline(loa_high_ci[0], color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax_scatter.axhline(loa_high_ci[1], color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax_scatter.axhline(loa_low_ci[0], color='green', linestyle=':', alpha=0.5, linewidth=1)
    ax_scatter.axhline(loa_low_ci[1], color='green', linestyle=':', alpha=0.5, linewidth=1)

    ax_scatter.set_xlabel(xlabel, fontsize=10)
    ax_scatter.set_ylabel(ylabel, fontsize=10)
    ax_scatter.set_title(title, fontsize=11, fontweight='bold')
    ax_scatter.grid(True, linestyle=':', alpha=0.6)

    ax_hist.hist(diff, bins=12, orientation='horizontal', color='#a2c4c9', edgecolor='black', alpha=0.7, density=True)
    
    if len(diff) > 3 and np.std(diff) > 0:
        kde = stats.gaussian_kde(diff)
        y_grid = np.linspace(min(diff) - 0.1*abs(min(diff)), max(diff) + 0.1*abs(max(diff)), 100)
        ax_hist.plot(kde(y_grid), y_grid, color='red', linewidth=1.5)

    ax_hist.axhline(bias, color='red', linestyle='-', linewidth=1.5)
    ax_hist.axhline(bias_ci[0], color='red', linestyle=':', alpha=0.6)
    ax_hist.axhline(bias_ci[1], color='red', linestyle=':', alpha=0.6)
    
    ax_hist.axhline(loa_high, color='green', linestyle='--', linewidth=1.5)
    ax_hist.axhline(loa_high_ci[0], color='green', linestyle=':', alpha=0.5)
    ax_hist.axhline(loa_high_ci[1], color='green', linestyle=':', alpha=0.5)
    
    ax_hist.axhline(loa_low, color='green', linestyle='--', linewidth=1.5)
    ax_hist.axhline(loa_low_ci[0], color='green', linestyle=':', alpha=0.5)
    ax_hist.axhline(loa_low_ci[1], color='green', linestyle=':', alpha=0.5)

    ax_hist.set_xlabel('Tần suất', fontsize=9)
    plt.setp(ax_hist.get_yticklabels(), visible=False)
    ax_hist.grid(True, linestyle=':', alpha=0.6)

    handles = [l_bias, l_loa_h, l_loa_l]
    labels = [h.get_label() for h in handles]
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.88, 0.5), fontsize=8.5, frameon=True)

    fig.subplots_adjust(left=0.1, right=0.85, top=0.9, bottom=0.1)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# ==============================================================================
# BƯỚC 1: TIỀN XỬ LÝ & TẠO 2 BỘ DỮ LIỆU CƠ SỞ
# ==============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📁 Thư mục lưu kết quả phân tích: '{OUTPUT_DIR}'")

df_raw = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df_raw.columns = df_raw.columns.astype(str).str.strip()

df_full = df_raw.copy()

req_cols = []
for req_analyte in REQUIRED_6_ANALYTES:
    if req_analyte in ANALYTE_PAIRS:
        col_a, col_b = ANALYTE_PAIRS[req_analyte]
        req_cols.extend([col_a, col_b])

existing_req_cols = [c for c in req_cols if c in df_raw.columns]
df_filtered_6 = df_raw.dropna(subset=existing_req_cols).copy()

print("="*70)
print(f"📊 BỘ DỮ LIỆU GỐC (FULL): Tổng số dòng = {len(df_full)}")
print(f"📊 BỘ DỮ LIỆU LỌC (ĐỦ 6 CHỈ SỐ): Tổng số dòng = {len(df_filtered_6)}")
print("="*70)

DATASETS = {
    'Full Data': df_full,
    'Filtered 6 Analytes': df_filtered_6
}

# ==============================================================================
# BƯỚC 2: THỰC HIỆN PHÂN TÍCH CHO CẢ 2 BỘ DỮ LIỆU VÀ CÁC PHƯƠNG PHÁP THAM CHIẾU
# ==============================================================================
abs_diff_summary = []
pct_diff_summary = []
pb_summary_results = []
pb_prediction_results = []
detected_outliers_records = []

REF_MODES = [
    {'code': 'PPA', 'label': 'Phương pháp Tham chiếu PPA'},
    {'code': 'Mean', 'label': 'Trung bình 2 phương pháp (PPA+PPB)/2'}
]

for dataset_name, df_data in DATASETS.items():
    print(f"\n🚀 Đang xử lý tập dữ liệu cơ sở: [{dataset_name}] ...")

    for analyte, (col_a, col_b) in ANALYTE_PAIRS.items():
        if col_a not in df_data.columns or col_b not in df_data.columns:
            continue

        df_analyte = df_data.dropna(subset=[col_a, col_b]).copy()
        if len(df_analyte) == 0:
            continue

        ppa = df_analyte[col_a].values
        ppb = df_analyte[col_b].values

        diff_types = ['Abs'] if analyte in INTEGER_ANALYTES else ['Abs', 'Pct']

        for ref_info in REF_MODES:
            ref_code = ref_info['code']
            ref_label = ref_info['label']

            for diff_mode in diff_types:
                is_pct = (diff_mode == 'Pct')
                unit_str = "%" if is_pct else ""

                if not is_pct:
                    diff_raw = ppb - ppa
                else:
                    if ref_code == 'PPA':
                        diff_raw = np.where(ppa != 0, ((ppb - ppa) / ppa) * 100.0, 0.0)
                    else: # Mean: (PPA + PPB) / 2
                        mean_raw = (ppa + ppb) / 2.0
                        diff_raw = np.where(mean_raw != 0, ((ppb - ppa) / mean_raw) * 100.0, 0.0)

                valid_mask = np.isfinite(diff_raw)
                diff_v = diff_raw[valid_mask]
                ppa_v = ppa[valid_mask]
                ppb_v = ppb[valid_mask]
                df_analyte_valid = df_analyte.iloc[valid_mask].copy()

                is_int_analyte = (analyte in INTEGER_ANALYTES)
                
                variants = get_analyte_variants(ppa_v, ppb_v, diff_v, df_analyte_valid, is_int_analyte)

                for var in variants:
                    ds_var_name = f"{dataset_name}{var['suffix']}"
                    safe_var_suffix = var['suffix'].replace(' [', '_').replace(']', '').replace(' ', '_')

                    v_ppa = var['ppa']
                    v_ppb = var['ppb']
                    v_diff = var['diff']
                    is_normal = var['is_normal']
                    p_shapiro = var['p_shapiro']
                    norm_desc = var['norm_desc']
                    n_detected = var['n_detected']
                    n_removed = var['n_removed']
                    outlier_idx_list = var['outlier_idx']

                    v_x_ref = v_ppa if ref_code == 'PPA' else (v_ppa + v_ppb) / 2.0
                    
                    # 🎯 SỬA TRỤC HOÀNH BIỂU ĐỒ KHÁC BIỆT: Bỏ tên chỉ số trước PPA/PPB
                    x_axis_label = "Phương pháp A (PPA)" if ref_code == 'PPA' else "Trung bình [PPA + PPB] / 2"

                    # --- 1. GHI NHẬN DANH SÁCH SỐ LẠC ---
                    if n_detected > 0 and n_removed == 0:
                        for out_i in outlier_idx_list:
                            outlier_row_dict = df_analyte_valid.iloc[out_i].to_dict()
                            meta_info = {
                                'Tập Dữ Liệu Cơ Sở': dataset_name,
                                'Chỉ số': analyte,
                                'Phương pháp Tham chiếu': ref_label,
                                'Loại Khác Biệt': 'Tuyệt đối (Abs)' if not is_pct else 'Tỷ lệ (%)',
                                'Giá trị PPA': ppa_v[out_i],
                                'Giá trị PPB': ppb_v[out_i],
                                'Giá trị Tham chiếu (X)': ppa_v[out_i] if ref_code == 'PPA' else (ppa_v[out_i] + ppb_v[out_i]) / 2.0,
                                'Giá trị Khác biệt': diff_v[out_i],
                                'Trạng thái Phân tích': f'Phát hiện {n_detected} số lạc -> Phân tích song song tập Gốc & tập Loại số lạc'
                            }
                            combined_outlier_info = {**meta_info, **outlier_row_dict}
                            detected_outliers_records.append(combined_outlier_info)

                    # --- 2. PASSING-BABLOK REGRESSION ---
                    if diff_mode == 'Abs' and ref_code == 'PPA':
                        target_ppa = PPA_PREDICT_VALUES.get(analyte, None)
                        slope, intercept, slope_ci, intercept_ci, pred_res_list = passing_bablok_fit_and_predict(
                            v_ppa, v_ppb, targets_x=target_ppa, n_boot=1000
                        )
                        
                        r_val, _ = stats.pearsonr(v_ppa, v_ppb)
                        has_constant_error = not (intercept_ci[0] <= 0 <= intercept_ci[1])
                        has_proportional_error = not (slope_ci[0] <= 1 <= slope_ci[1])
                        
                        pb_interpretation = []
                        if not has_constant_error and not has_proportional_error:
                            pb_interpretation.append("Không có sai số cố định và không có sai số tỷ lệ.")
                        else:
                            if has_constant_error:
                                pb_interpretation.append(f"CÓ SAI SỐ CỐ ĐỊNH (Intercept 95% CI [{fmt(intercept_ci[0])}, {fmt(intercept_ci[1])}] không chứa 0).")
                            if has_proportional_error:
                                pb_interpretation.append(f"CÓ SAI SỐ TỶ LỆ (Slope 95% CI [{fmt(slope_ci[0])}, {fmt(slope_ci[1])}] không chứa 1).")

                        pb_summary_results.append({
                            'Tập Dữ Liệu': ds_var_name,
                            'Chỉ số': analyte,
                            'N phân tích': len(v_ppa),
                            'Pearson R': fmt(r_val, 3),
                            'Phương trình Hồi quy': f"PPB = {fmt(intercept)} + {fmt(slope)} * PPA",
                            'Hằng số Intercept A': fmt(intercept),
                            '95% CI Intercept': f"[{fmt(intercept_ci[0])}, {fmt(intercept_ci[1])}]",
                            'Có sai số cố định?': 'Có' if has_constant_error else 'Không',
                            'Hệ số Slope B': fmt(slope),
                            '95% CI Slope': f"[{fmt(slope_ci[0])}, {fmt(slope_ci[1])}]",
                            'Có sai số tỷ lệ?': 'Có' if has_proportional_error else 'Không',
                            'Phiên giải Kết quả': " ".join(pb_interpretation)
                        })

                        for pred_res in pred_res_list:
                            px = pred_res['target_x']
                            py = pred_res['y_pred']
                            b_abs_pb = pred_res['bias_abs']
                            b_abs_ci_pb = pred_res['bias_abs_ci']
                            b_pct_pb = pred_res['bias_pct']
                            b_pct_ci_pb = pred_res['bias_pct_ci']

                            levels = ['Optimal', 'Desirable', 'Minimum']
                            clsi_pb_evals = {}

                            for lvl in levels:
                                b_max_pct = BIAS_LIMITS.get(analyte, {}).get('Pct', {}).get(lvl, None)
                                b_max_abs = BIAS_LIMITS.get(analyte, {}).get('Abs', {}).get(lvl, None)

                                if b_max_pct is not None:
                                    code, meaning = classify_clsi_bias(b_pct_pb, b_pct_ci_pb[0], b_pct_ci_pb[1], b_max_pct)
                                    b_max_str = f"{fmt(b_max_pct)}%"
                                elif b_max_abs is not None:
                                    code, meaning = classify_clsi_bias(b_abs_pb, b_abs_ci_pb[0], b_abs_ci_pb[1], b_max_abs)
                                    b_max_str = fmt(b_max_abs)
                                else:
                                    code, meaning = "N/A", "Không khai báo B_max"
                                    b_max_str = "N/A"

                                clsi_pb_evals[f'B_max ({lvl})'] = b_max_str
                                clsi_pb_evals[f'Phân loại ({lvl})'] = code
                                clsi_pb_evals[f'Diễn giải ({lvl})'] = meaning

                            row_pred = {
                                'Tập Dữ Liệu': ds_var_name,
                                'Chỉ số': analyte,
                                'Giá trị PPA Nhập vào': fmt(px),
                                'PPB Ước tính (Bootstrap)': fmt(py),
                                '95% CI của PPB Ước tính': f"[{fmt(pred_res['y_pred_ci'][0])}, {fmt(pred_res['y_pred_ci'][1])}]",
                                'Độ lệch Tuyệt đối (PPB - PPA)': fmt(b_abs_pb),
                                '95% CI Độ lệch Tuyệt đối': f"[{fmt(b_abs_ci_pb[0])}, {fmt(b_abs_ci_pb[1])}]",
                                'Độ lệch Tỷ lệ (%)': f"{fmt(b_pct_pb)}%",
                                '95% CI Độ lệch Tỷ lệ (%)': f"[{fmt(b_pct_ci_pb[0])}%, {fmt(b_pct_ci_pb[1])}%]",
                            }

                            for lvl in levels:
                                row_pred[f'Ngưỡng B_max ({lvl})'] = clsi_pb_evals[f'B_max ({lvl})']
                                row_pred[f'Phân loại CLSI ({lvl})'] = clsi_pb_evals[f'Phân loại ({lvl})']
                                row_pred[f'Diễn giải Lâm sàng ({lvl})'] = clsi_pb_evals[f'Diễn giải ({lvl})']

                            pb_prediction_results.append(row_pred)

                        fig, ax0 = plt.subplots(figsize=(9.5, 6.5), dpi=300)
                        ax0.scatter(v_ppa, v_ppb, color='#2b5c8f', alpha=0.6, edgecolors='k', label='Giá trị khảo sát', zorder=3)
                        
                        all_x = list(v_ppa) + [p['target_x'] for p in pred_res_list]
                        all_y = list(v_ppb) + [p['y_pred'] for p in pred_res_list]
                        
                        raw_x_min, raw_x_max = min(all_x), max(all_x)
                        raw_y_min, raw_y_max = min(all_y), max(all_y)

                        span_x = raw_x_max - raw_x_min if raw_x_max != raw_x_min else 1.0
                        span_y = raw_y_max - raw_y_min if raw_y_max != raw_y_min else 1.0

                        margin_pct = 0.12
                        plot_xlim = (raw_x_min - margin_pct * span_x, raw_x_max + margin_pct * span_x)
                        plot_ylim = (raw_y_min - margin_pct * span_y, raw_y_max + margin_pct * span_y)

                        x_start = max(plot_xlim[0], (plot_ylim[0] - intercept) / slope if slope != 0 else plot_xlim[0])
                        x_end = min(plot_xlim[1], (plot_ylim[1] - intercept) / slope if slope != 0 else plot_xlim[1])
                        x_vals = np.linspace(x_start, x_end, 200)

                        x_diag_start = max(plot_xlim[0], plot_ylim[0])
                        x_diag_end = min(plot_xlim[1], plot_ylim[1])
                        x_diag_vals = np.linspace(x_diag_start, x_diag_end, 200)
                        ax0.plot(x_diag_vals, x_diag_vals, 'k--', label='Y=X', alpha=0.7, zorder=1)
                        
                        lbl_pb = f'Y = {fmt(intercept)} + {fmt(slope)}X'
                        ax0.plot(x_vals, intercept + slope * x_vals, 'r-', label=lbl_pb, linewidth=1.8, zorder=2)
                        ax0.plot([], [], ' ', label=f'R = {fmt(r_val, 3)}')

                        ax0.set_xlim(plot_xlim)
                        ax0.set_ylim(plot_ylim)

                        sorted_pred_res = sorted(pred_res_list, key=lambda x: x['target_x'])

                        for idx, pred_res in enumerate(sorted_pred_res):
                            px = pred_res['target_x']
                            py = pred_res['y_pred']
                            p_bias_abs = pred_res['bias_abs']
                            p_bias_pct = pred_res['bias_pct']

                            lbl_point = 'Giá trị dự báo tại các điểm\nquyết định lâm sàng' if idx == 0 else ""
                            ax0.scatter([px], [py], color='red', s=80, zorder=5, label=lbl_point, edgecolors='black')
                            
                            ax0.vlines(x=px, ymin=plot_ylim[0], ymax=py, color='gray', linestyle=':', alpha=0.5)
                            ax0.hlines(y=py, xmin=plot_xlim[0], xmax=px, color='gray', linestyle=':', alpha=0.5)

                            offset_x = span_x * 0.08
                            if len(sorted_pred_res) > 1 and idx < len(sorted_pred_res) - 1:
                                if (sorted_pred_res[idx+1]['target_x'] - px) < (span_x * 0.15):
                                    offset_y = (idx % 2 - 0.5) * (span_y * 0.12)
                                else:
                                    offset_y = 0
                            else:
                                offset_y = 0

                            text_x = px + offset_x
                            text_y = py + offset_y

                            anno_text = f"X={fmt(px)}, Y={fmt(py)}\nBias={fmt(p_bias_abs)} ({fmt(p_bias_pct)}%)"
                            
                            ax0.annotate(
                                anno_text, xy=(px, py), xytext=(text_x, text_y),
                                arrowprops=dict(
                                    arrowstyle="->",
                                    connectionstyle="arc3,rad=0",
                                    color="black",
                                    lw=1.0
                                ),
                                fontsize=8,
                                fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.4", fc="#ffffcc", ec="#cc9900", alpha=0.9),
                                ha='left', va='center', zorder=6
                            )

                        # 🎯 SỬA CÁC TRỤC VÀ TIÊU ĐỀ CHO PASSING-BABLOK
                        ax0.set_xlabel('Phương pháp A (PPA)', fontsize=10)
                        ax0.set_ylabel('Phương pháp B (PPB)', fontsize=10)
                        ax0.set_title(f'Passing-Bablok Regression - {analyte}', fontsize=11, fontweight='bold')
                        ax0.grid(True, linestyle=':', alpha=0.6)

                        max_ppa_data = np.max(v_ppa)
                        max_ppb_data = np.max(v_ppb)
                        min_ppa_data = np.min(v_ppa)
                        min_ppb_data = np.min(v_ppb)

                        if (raw_x_max / (max_ppa_data + 1e-5)) > 2.0:
                            ax_inset = fig.add_axes([0.18, 0.60, 0.30, 0.25]) 
                            
                            span_in_x = max_ppa_data - min_ppa_data if max_ppa_data != min_ppa_data else 1.0
                            span_in_y = max_ppb_data - min_ppb_data if max_ppb_data != min_ppb_data else 1.0
                            
                            inset_xlim = (min_ppa_data - 0.08 * span_in_x, max_ppa_data + 0.08 * span_in_x)
                            inset_ylim = (min_ppb_data - 0.08 * span_in_y, max_ppb_data + 0.08 * span_in_y)

                            ax_inset.scatter(v_ppa, v_ppb, color='#2b5c8f', alpha=0.6, edgecolors='k', s=20, zorder=3)
                            
                            x_inset_line = np.linspace(inset_xlim[0], inset_xlim[1], 100)
                            ax_inset.plot(x_inset_line, x_inset_line, 'k--', alpha=0.5)
                            ax_inset.plot(x_inset_line, intercept + slope * x_inset_line, 'r-', linewidth=1.5)

                            ax_inset.set_xlim(inset_xlim)
                            ax_inset.set_ylim(inset_ylim)
                            
                            ax_inset.set_title('Dữ liệu khảo sát', fontsize=8.5, fontweight='bold', color='darkblue', y=1.03, pad=2)
                            ax_inset.tick_params(labelsize=7)
                            ax_inset.grid(True, linestyle=':', alpha=0.5)

                        handles0, labels0 = ax0.get_legend_handles_labels()
                        ax0.legend(handles0, labels0, loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=True, labelspacing=0.8)

                        fig.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)
                        
                        prefix_file = "PB_Full" if dataset_name == 'Full Data' else "PB_Filtered"
                        plt.savefig(os.path.join(OUTPUT_DIR, f'{prefix_file}{safe_var_suffix}_{analyte}.png'), dpi=300, bbox_inches='tight')
                        plt.close()

                    # --- 3. BLAND-ALTMAN & KIỂM ĐỊNH THỐNG KÊ ---
                    n_final = len(v_diff)

                    if is_int_analyte:
                        data_type = "Số nguyên (Non-parametric Bootstrap)"
                        bias_stat_used = "Median"
                        b = np.median(v_diff)
                        boots = [np.median(np.random.choice(v_diff, size=n_final, replace=True)) for _ in range(1000)]
                        c_low, c_high = np.percentile(boots, [2.5, 97.5])
                        l_low, l_high = np.percentile(v_diff, [2.5, 97.5])
                        
                        boot_loas_low = [np.percentile(np.random.choice(v_diff, size=n_final, replace=True), 2.5) for _ in range(1000)]
                        boot_loas_high = [np.percentile(np.random.choice(v_diff, size=n_final, replace=True), 97.5) for _ in range(1000)]
                        l_low_ci = np.percentile(boot_loas_low, [2.5, 97.5])
                        l_high_ci = np.percentile(boot_loas_high, [2.5, 97.5])
                    else:
                        if is_normal:
                            data_type = "Phân bố chuẩn (Parametric - Mean ± 1.96 SD)"
                            bias_stat_used = "Mean"
                            b = np.mean(v_diff)
                            sd = np.std(v_diff, ddof=1)
                            se_b = sd / np.sqrt(n_final)
                            c_low, c_high = b - 1.96 * se_b, b + 1.96 * se_b
                            
                            l_low, l_high = b - 1.96 * sd, b + 1.96 * sd
                            se_loa = np.sqrt(3 * (sd**2) / n_final)
                            l_low_ci = (l_low - 1.96 * se_loa, l_low + 1.96 * se_loa)
                            l_high_ci = (l_high - 1.96 * se_loa, l_high + 1.96 * se_loa)
                        else:
                            data_type = "Không chuẩn (Non-parametric Bootstrap)"
                            bias_stat_used = "Median"
                            b = np.median(v_diff)
                            boots = [np.median(np.random.choice(v_diff, size=n_final, replace=True)) for _ in range(1000)]
                            c_low, c_high = np.percentile(boots, [2.5, 97.5])
                            
                            l_low, l_high = np.percentile(v_diff, [2.5, 97.5])
                            boot_loas_low = [np.percentile(np.random.choice(v_diff, size=n_final, replace=True), 2.5) for _ in range(1000)]
                            boot_loas_high = [np.percentile(np.random.choice(v_diff, size=n_final, replace=True), 97.5) for _ in range(1000)]
                            l_low_ci = np.percentile(boot_loas_low, [2.5, 97.5])
                            l_high_ci = np.percentile(boot_loas_high, [2.5, 97.5])

                    ppa_m, ppa_s = np.mean(v_ppa), np.std(v_ppa, ddof=1)
                    ppb_m, ppb_s = np.mean(v_ppb), np.std(v_ppb, ddof=1)
                    ppa_med, ppa_iqr = np.median(v_ppa), np.percentile(v_ppa, 75) - np.percentile(v_ppa, 25)
                    ppb_med, ppb_iqr = np.median(v_ppb), np.percentile(v_ppb, 75) - np.percentile(v_ppb, 25)

                    test_name, test_stat, p_val_diff, sig_conclusion = perform_difference_test(v_ppa, v_ppb, is_normal)

                    prefix_ba = "BA_Full" if dataset_name == 'Full Data' else "BA_Filtered"
                    file_suffix = f"{prefix_ba}_Ref{ref_code}{safe_var_suffix}_Abs" if not is_pct else f"{prefix_ba}_Ref{ref_code}{safe_var_suffix}_Pct"
                    
                    if not is_pct:
                        y_label_str = "Khác biệt tuyệt đối (PPB - PPA)"
                    else:
                        if ref_code == 'PPA':
                            y_label_str = "Khác biệt tỷ lệ % [(PPB - PPA) / PPA]"
                        else:
                            y_label_str = "Khác biệt tỷ lệ % [(PPB-PPA)/(PPA+PPB)/2]"

                    # 🎯 SỬA TIÊU ĐỀ BIỂU ĐỒ KHÁC BIỆT: Bỏ chú giải sau tên chỉ số
                    chart_title = f"Bland-Altman Plot ({'Tuyệt đối' if not is_pct else 'Tỷ lệ %'}) - {analyte}"
                    fig_out_path = os.path.join(OUTPUT_DIR, f"{file_suffix}_{analyte}.png")

                    plot_bland_altman_with_histogram(
                        v_x_ref, v_diff, b, (c_low, c_high), l_low, l_low_ci, l_high, l_high_ci, 
                        chart_title, x_axis_label, y_label_str, unit_str, fig_out_path
                    )

                    eval_key = 'Pct' if is_pct else 'Abs'
                    levels = ['Optimal', 'Desirable', 'Minimum']
                    clsi_evals = {}

                    for lvl in levels:
                        b_max_val = BIAS_LIMITS.get(analyte, {}).get(eval_key, {}).get(lvl, None)
                        code, meaning = classify_clsi_bias(b, c_low, c_high, b_max_val)
                        clsi_evals[f'B_max ({lvl})'] = f"{fmt(b_max_val)}{unit_str}" if b_max_val is not None else "N/A"
                        clsi_evals[f'Phân loại ({lvl})'] = code
                        clsi_evals[f'Diễn giải ({lvl})'] = meaning

                    row_dict = {
                        'Tập Dữ Liệu': ds_var_name,
                        'Phương pháp Tham chiếu (X-axis / %)': ref_label,
                        'Chỉ số': analyte,
                        'N gốc': len(diff_v),
                        'Shapiro-Wilk p-value': fmt(p_shapiro, 3) if isinstance(p_shapiro, float) else p_shapiro,
                        'Kết quả Phân bố & Loại Lạc': norm_desc,
                        'Số lượng số lạc phát hiện (GESD)': n_detected,
                        'Số lượng số lạc đã loại bỏ': n_removed,
                        'N phân tích cuối': len(v_diff),
                        'Phương pháp Thống kê': data_type,
                        'PPA Mean ± SD': f"{fmt(ppa_m)} ± {fmt(ppa_s)}",
                        'PPB Mean ± SD': f"{fmt(ppb_m)} ± {fmt(ppb_s)}",
                        'PPA Median (IQR) [Min - Max]': f"{fmt(ppa_med)} ({fmt(ppa_iqr)}) [{fmt(np.min(v_ppa))} - {fmt(np.max(v_ppa))}]",
                        'PPB Median (IQR) [Min - Max]': f"{fmt(ppb_med)} ({fmt(ppb_iqr)}) [{fmt(np.min(v_ppb))} - {fmt(np.max(v_ppb))}]",
                        
                        'Phương pháp Kiểm định Khác biệt': test_name,
                        'Giá trị Thống kê (Statistic)': fmt(test_stat, 3),
                        'P-value (Khác biệt)': fmt(p_val_diff, 4),
                        'Kết luận Ý nghĩa Thống kê': sig_conclusion,

                        'Loại Giá trị Bias Đang dùng': bias_stat_used,
                        'Độ lệch (Bias)': f"{fmt(b)}{unit_str}",
                        '95% CI của Bias': f"[{fmt(c_low)}, {fmt(c_high)}]{unit_str}",
                        'Lower LoA': f"{fmt(l_low)}{unit_str}",
                        'Upper LoA': f"{fmt(l_high)}{unit_str}"
                    }

                    for lvl in levels:
                        row_dict[f'Ngưỡng B_max ({lvl})'] = clsi_evals[f'B_max ({lvl})']
                        row_dict[f'Phân loại CLSI ({lvl})'] = clsi_evals[f'Phân loại ({lvl})']
                        row_dict[f'Diễn giải Lâm sàng ({lvl})'] = clsi_evals[f'Diễn giải ({lvl})']

                    if not is_pct:
                        abs_diff_summary.append(row_dict)
                    else:
                        pct_diff_summary.append(row_dict)

# ==============================================================================
# BƯỚC 3: XUẤT BÁO CÁO EXCEL TỔNG HỢP CÓ SHEET DÀNH RIÊNG CHO SỐ LẠC
# ==============================================================================
df_abs_summary = pd.DataFrame(abs_diff_summary)
df_pct_summary = pd.DataFrame(pct_diff_summary)
df_pb_summary = pd.DataFrame(pb_summary_results)
df_pb_predict = pd.DataFrame(pb_prediction_results)
df_outliers_summary = pd.DataFrame(detected_outliers_records)

excel_out = os.path.join(OUTPUT_DIR, f'Bao_Cao_So_Sanh_Hai_Bo_Du_Lieu_{TIMESTAMP}.xlsx')

with pd.ExcelWriter(excel_out, engine='openpyxl') as writer:
    if not df_abs_summary.empty:
        df_abs_summary.to_excel(writer, sheet_name='Khác biệt tuyệt đối (Abs)', index=False)
    if not df_pct_summary.empty:
        df_pct_summary.to_excel(writer, sheet_name='Khác biệt tỷ lệ (Pct)', index=False)
    if not df_pb_summary.empty:
        df_pb_summary.to_excel(writer, sheet_name='Passing_Bablok_Regression', index=False)
    if not df_pb_predict.empty:
        df_pb_predict.to_excel(writer, sheet_name='Passing_Bablok_Prediction', index=False)
    if not df_outliers_summary.empty:
        df_outliers_summary.to_excel(writer, sheet_name='Danh_Sach_So_Lac', index=False)

print("="*70)
print(f"🎉 CHƯƠNG TRÌNH ĐÃ HOÀN THÀNH!")
print(f"📌 Đã cập nhật xong định dạng nhãn & tiêu đề:")
print(f"   • Trục hoành Bland-Altman: 'Phương pháp A (PPA)' hoặc 'Trung bình [PPA + PPB] / 2'")
print(f"   • Trục hoành/tung Passing-Bablok: 'Phương pháp A (PPA)', 'Phương pháp B (PPB)'")
print(f"   • Tiêu đề biểu đồ: Rút gọn chỉ còn tên biểu đồ + tên chỉ số")
print(f"📁 Kết quả Excel và Biểu đồ lưu tại thư mục: '{OUTPUT_DIR}'")
print("="*70)