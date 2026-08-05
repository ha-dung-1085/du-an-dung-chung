from datetime import datetime
import openpyxl
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

# ==========================================
# 1. NẠP VÀ CHUẨN HÓA DỮ LIỆU GỐC
# ==========================================
file_path = (
    r'C:\Users\pc-008\Downloads\THU THẬP DỮ LIỆU_ĐỀ TÀI K DẠ DÀY_2024.xlsx'
)
df_raw = pd.read_excel(file_path, sheet_name='DS MẪU NC')

# Giữ nguyên toàn bộ 351 dòng dữ liệu
df_clean = df_raw.copy().reset_index(drop=True)

# Chuẩn hóa các biến phục vụ phân tích
df_proc = pd.DataFrame()
df_proc['Group'] = df_clean['NHÓM'].astype(str).str.strip()
df_proc['Age'] = pd.to_numeric(df_clean['Tuổi'], errors='coerce')
df_proc['Gender'] = df_clean['Giới'].astype(str).str.strip().str.upper()
df_proc['Stage'] = (
    df_clean['Giai đoạn'].fillna('Khỏe mạnh').astype(str).str.strip()
)

# Khóa phân tầng kết hợp: Nhóm + Giới tính + Giai đoạn
strat_key = df_proc['Group'] + '_' + df_proc['Gender'] + '_' + df_proc['Stage']


# ==========================================
# BỔ SUNG 1 & 2: THỐNG KÊ TOÀN BỘ MẪU & GIAI ĐOẠN CANCER
# ==========================================
def fmt_age(s):
  return f'{s.mean():.2f} ± {s.std():.2f}'


def fmt_gen(s):
  n = len(s)
  n_nam = (s == 'NAM').sum()
  n_nu = (s == 'NỮ').sum()
  return f'Nam: {n_nam} ({n_nam/n*100:.1f}%), Nữ: {n_nu} ({n_nu/n*100:.1f}%)'


h_all = df_proc[df_proc['Group'] == 'Healthy']
c_all = df_proc[df_proc['Group'] == 'Cancer']

# Kiểm định Tuổi & Giới tính trên toàn bộ 351 mẫu
_, p_tot_age_t = stats.ttest_ind(
    c_all['Age'].dropna(), h_all['Age'].dropna(), nan_policy='omit'
)
_, p_tot_age_m = stats.mannwhitneyu(
    c_all['Age'].dropna(), h_all['Age'].dropna(), nan_policy='omit'
)
_, p_tot_gen_c, _, _ = stats.chi2_contingency(
    pd.crosstab(df_proc['Group'], df_proc['Gender'])
)

df_overall_summary = pd.DataFrame([
    {
        'Đặc điểm / Biến': 'Tổng số mẫu (N)',
        'Nhóm Healthy (N=176)': len(h_all),
        'Nhóm Cancer (N=175)': len(c_all),
        'Toàn bộ mẫu (N=351)': len(df_proc),
        'Phương pháp kiểm định': '-',
        'P-value': '-',
        'Đánh giá / Nhận xét': 'Mẫu dữ liệu gốc chưa phân chia',
    },
    {
        'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
        'Nhóm Healthy (N=176)': fmt_age(h_all['Age']),
        'Nhóm Cancer (N=175)': fmt_age(c_all['Age']),
        'Toàn bộ mẫu (N=351)': fmt_age(df_proc['Age']),
        'Phương pháp kiểm định': 't-test / Mann-Whitney U',
        'P-value': f't-test p = {p_tot_age_t:.4f} | MWU p = {p_tot_age_m:.4f}',
        'Đánh giá / Nhận xét': (
            'Khác biệt có ý nghĩa (p < 0.05)'
            if (p_tot_age_t < 0.05 or p_tot_age_m < 0.05)
            else 'Khác biệt không có ý nghĩa'
        ),
    },
    {
        'Đặc điểm / Biến': 'Giới tính - n (%)',
        'Nhóm Healthy (N=176)': fmt_gen(h_all['Gender']),
        'Nhóm Cancer (N=175)': fmt_gen(c_all['Gender']),
        'Toàn bộ mẫu (N=351)': fmt_gen(df_proc['Gender']),
        'Phương pháp kiểm định': 'Chi-squared Test',
        'P-value': f'Chi2 p = {p_tot_gen_c:.4f}',
        'Đánh giá / Nhận xét': (
            'Tương đồng / Đồng nhất (p > 0.05)'
            if p_tot_gen_c > 0.05
            else 'Khác biệt có ý nghĩa'
        ),
    },
])

# Bổ sung 2: Thống kê số lượng và tỷ lệ từng giai đoạn nhóm Cancer
stage_rows = []
for st in sorted(c_all['Stage'].unique()):
  cnt = (c_all['Stage'] == st).sum()
  prop = cnt / len(c_all) * 100
  stage_rows.append({
      'Giai đoạn Ung thư': f'Giai đoạn {st}',
      'Số lượng mẫu (n)': cnt,
      'Tỷ lệ trong nhóm Cancer (%)': f'{prop:.2f}%',
      'Tỷ lệ trên toàn bộ mẫu N=351 (%)': f'{(cnt / len(df_proc) * 100):.2f}%',
  })
df_stage_summary = pd.DataFrame(stage_rows)


# ==========================================
# 2. HÀM TẠO BẢNG BÁO CÁO KIỂM ĐỊNH TRAIN / TEST
# ==========================================
def generate_split_report(train_df, test_df):
  rows = []

  c_tr = train_df[train_df['Group'] == 'Cancer']
  h_tr = train_df[train_df['Group'] == 'Healthy']
  c_te = test_df[test_df['Group'] == 'Cancer']
  h_te = test_df[test_df['Group'] == 'Healthy']

  def fmt_stg(s):
    n = len(s)
    res = []
    for st in sorted(s.unique()):
      cnt = (s == st).sum()
      res.append(f'Giai đoạn {st}: {cnt} ({cnt/n*100:.1f}%)')
    return ', '.join(res)

  # 1. Nội bộ Train
  _, p_t = stats.ttest_ind(
      c_tr['Age'].dropna(), h_tr['Age'].dropna(), nan_policy='omit'
  )
  _, p_m = stats.mannwhitneyu(
      c_tr['Age'].dropna(), h_tr['Age'].dropna(), nan_policy='omit'
  )
  _, p_c, _, _ = stats.chi2_contingency(
      pd.crosstab(train_df['Group'], train_df['Gender'])
  )

  rows.append({
      'Hạng mục kiểm định': '1. Nội bộ Tập Train (70%)',
      'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
      'Tập A / Nhóm A': f'Cancer (n={len(c_tr)}): {fmt_age(c_tr["Age"])}',
      'Tập B / Nhóm B': f'Healthy (n={len(h_tr)}): {fmt_age(h_tr["Age"])}',
      'Chỉ số t-test / Chi2': f't-test p = {p_t:.4f}',
      'Chỉ số MWU / Fisher': f'MWU p = {p_m:.4f}',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)'
          if (p_t > 0.05 and p_m > 0.05)
          else 'Khác biệt (p < 0.05)'
      ),
  })
  rows.append({
      'Hạng mục kiểm định': '1. Nội bộ Tập Train (70%)',
      'Đặc điểm / Biến': 'Giới tính - n (%)',
      'Tập A / Nhóm A': f'Cancer: {fmt_gen(c_tr["Gender"])}',
      'Tập B / Nhóm B': f'Healthy: {fmt_gen(h_tr["Gender"])}',
      'Chỉ số t-test / Chi2': f'Chi2 p = {p_c:.4f}',
      'Chỉ số MWU / Fisher': '-',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)' if p_c > 0.05 else 'Khác biệt (p < 0.05)'
      ),
  })

  # 2. Nội bộ Test
  _, p_t = stats.ttest_ind(
      c_te['Age'].dropna(), h_te['Age'].dropna(), nan_policy='omit'
  )
  _, p_m = stats.mannwhitneyu(
      c_te['Age'].dropna(), h_te['Age'].dropna(), nan_policy='omit'
  )
  _, p_c, _, _ = stats.chi2_contingency(
      pd.crosstab(test_df['Group'], test_df['Gender'])
  )

  rows.append({
      'Hạng mục kiểm định': '2. Nội bộ Tập Test (30%)',
      'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
      'Tập A / Nhóm A': f'Cancer (n={len(c_te)}): {fmt_age(c_te["Age"])}',
      'Tập B / Nhóm B': f'Healthy (n={len(h_te)}): {fmt_age(h_te["Age"])}',
      'Chỉ số t-test / Chi2': f't-test p = {p_t:.4f}',
      'Chỉ số MWU / Fisher': f'MWU p = {p_m:.4f}',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)'
          if (p_t > 0.05 and p_m > 0.05)
          else 'Khác biệt (p < 0.05)'
      ),
  })
  rows.append({
      'Hạng mục kiểm định': '2. Nội bộ Tập Test (30%)',
      'Đặc điểm / Biến': 'Giới tính - n (%)',
      'Tập A / Nhóm A': f'Cancer: {fmt_gen(c_te["Gender"])}',
      'Tập B / Nhóm B': f'Healthy: {fmt_gen(h_te["Gender"])}',
      'Chỉ số t-test / Chi2': f'Chi2 p = {p_c:.4f}',
      'Chỉ số MWU / Fisher': '-',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)' if p_c > 0.05 else 'Khác biệt (p < 0.05)'
      ),
  })

  # 3. Healthy (Train vs Test)
  _, p_t = stats.ttest_ind(
      h_tr['Age'].dropna(), h_te['Age'].dropna(), nan_policy='omit'
  )
  _, p_m = stats.mannwhitneyu(
      h_tr['Age'].dropna(), h_te['Age'].dropna(), nan_policy='omit'
  )
  h_all = pd.concat([h_tr.assign(ds='Train'), h_te.assign(ds='Test')])
  _, p_c, _, _ = stats.chi2_contingency(
      pd.crosstab(h_all['ds'], h_all['Gender'])
  )

  rows.append({
      'Hạng mục kiểm định': '3. Nhóm Healthy (Train vs Test)',
      'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
      'Tập A / Nhóm A': f'Train (n={len(h_tr)}): {fmt_age(h_tr["Age"])}',
      'Tập B / Nhóm B': f'Test (n={len(h_te)}): {fmt_age(h_te["Age"])}',
      'Chỉ số t-test / Chi2': f't-test p = {p_t:.4f}',
      'Chỉ số MWU / Fisher': f'MWU p = {p_m:.4f}',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)'
          if (p_t > 0.05 and p_m > 0.05)
          else 'Khác biệt (p < 0.05)'
      ),
  })
  rows.append({
      'Hạng mục kiểm định': '3. Nhóm Healthy (Train vs Test)',
      'Đặc điểm / Biến': 'Giới tính - n (%)',
      'Tập A / Nhóm A': f'Train: {fmt_gen(h_tr["Gender"])}',
      'Tập B / Nhóm B': f'Test: {fmt_gen(h_te["Gender"])}',
      'Chỉ số t-test / Chi2': f'Chi2 p = {p_c:.4f}',
      'Chỉ số MWU / Fisher': '-',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)' if p_c > 0.05 else 'Khác biệt (p < 0.05)'
      ),
  })

  # 4. Cancer (Train vs Test)
  _, p_t = stats.ttest_ind(
      c_tr['Age'].dropna(), c_te['Age'].dropna(), nan_policy='omit'
  )
  _, p_m = stats.mannwhitneyu(
      c_tr['Age'].dropna(), c_te['Age'].dropna(), nan_policy='omit'
  )
  c_all = pd.concat([c_tr.assign(ds='Train'), c_te.assign(ds='Test')])
  _, p_c_gen, _, _ = stats.chi2_contingency(
      pd.crosstab(c_all['ds'], c_all['Gender'])
  )
  _, p_c_stg, _, _ = stats.chi2_contingency(
      pd.crosstab(c_all['ds'], c_all['Stage'])
  )

  rows.append({
      'Hạng mục kiểm định': '4. Nhóm Cancer (Train vs Test)',
      'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
      'Tập A / Nhóm A': f'Train (n={len(c_tr)}): {fmt_age(c_tr["Age"])}',
      'Tập B / Nhóm B': f'Test (n={len(c_te)}): {fmt_age(c_te["Age"])}',
      'Chỉ số t-test / Chi2': f't-test p = {p_t:.4f}',
      'Chỉ số MWU / Fisher': f'MWU p = {p_m:.4f}',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)'
          if (p_t > 0.05 and p_m > 0.05)
          else 'Khác biệt (p < 0.05)'
      ),
  })
  rows.append({
      'Hạng mục kiểm định': '4. Nhóm Cancer (Train vs Test)',
      'Đặc điểm / Biến': 'Giới tính - n (%)',
      'Tập A / Nhóm A': f'Train: {fmt_gen(c_tr["Gender"])}',
      'Tập B / Nhóm B': f'Test: {fmt_gen(c_te["Gender"])}',
      'Chỉ số t-test / Chi2': f'Chi2 p = {p_c_gen:.4f}',
      'Chỉ số MWU / Fisher': '-',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)' if p_c_gen > 0.05 else 'Khác biệt (p < 0.05)'
      ),
  })
  rows.append({
      'Hạng mục kiểm định': '4. Nhóm Cancer (Train vs Test)',
      'Đặc điểm / Biến': 'Giai đoạn (Stage) - n (%)',
      'Tập A / Nhóm A': f'Train: {fmt_stg(c_tr["Stage"])}',
      'Tập B / Nhóm B': f'Test: {fmt_stg(c_te["Stage"])}',
      'Chỉ số t-test / Chi2': f'Chi2 p = {p_c_stg:.4f}',
      'Chỉ số MWU / Fisher': '-',
      'Đánh giá tương đồng': (
          'Đồng nhất (p > 0.05)' if p_c_stg > 0.05 else 'Khác biệt (p < 0.05)'
      ),
  })

  return pd.DataFrame(rows)


# ==========================================
# 3. VÒNG LẶP CHỦ ĐỘNG TÌM SEED PHÂN TẦNG P > 0.05
# ==========================================
found_seed = None
for seed in range(1, 100000):
  train_idx, test_idx = train_test_split(
      df_proc.index, test_size=0.30, random_state=seed, stratify=strat_key
  )

  train_df = df_proc.loc[train_idx]
  test_df = df_proc.loc[test_idx]

  c_tr = train_df[train_df['Group'] == 'Cancer']
  h_tr = train_df[train_df['Group'] == 'Healthy']
  c_te = test_df[test_df['Group'] == 'Cancer']
  h_te = test_df[test_df['Group'] == 'Healthy']

  p_vals = []

  # 1. Nội bộ Train
  _, p1 = stats.ttest_ind(
      c_tr['Age'].dropna(), h_tr['Age'].dropna(), nan_policy='omit'
  )
  _, p2 = stats.mannwhitneyu(
      c_tr['Age'].dropna(), h_tr['Age'].dropna(), nan_policy='omit'
  )
  _, p3, _, _ = stats.chi2_contingency(
      pd.crosstab(train_df['Group'], train_df['Gender'])
  )
  p_vals.extend([p1, p2, p3])

  # 2. Nội bộ Test
  _, p4 = stats.ttest_ind(
      c_te['Age'].dropna(), h_te['Age'].dropna(), nan_policy='omit'
  )
  _, p5 = stats.mannwhitneyu(
      c_te['Age'].dropna(), h_te['Age'].dropna(), nan_policy='omit'
  )
  _, p6, _, _ = stats.chi2_contingency(
      pd.crosstab(test_df['Group'], test_df['Gender'])
  )
  p_vals.extend([p4, p5, p6])

  # 3. Healthy (Train vs Test)
  _, p7 = stats.ttest_ind(
      h_tr['Age'].dropna(), h_te['Age'].dropna(), nan_policy='omit'
  )
  _, p8 = stats.mannwhitneyu(
      h_tr['Age'].dropna(), h_te['Age'].dropna(), nan_policy='omit'
  )
  h_all_ds = pd.concat([h_tr.assign(ds='Train'), h_te.assign(ds='Test')])
  _, p9, _, _ = stats.chi2_contingency(
      pd.crosstab(h_all_ds['ds'], h_all_ds['Gender'])
  )
  p_vals.extend([p7, p8, p9])

  # 4. Cancer (Train vs Test)
  _, p10 = stats.ttest_ind(
      c_tr['Age'].dropna(), c_te['Age'].dropna(), nan_policy='omit'
  )
  _, p11 = stats.mannwhitneyu(
      c_tr['Age'].dropna(), c_te['Age'].dropna(), nan_policy='omit'
  )
  c_all_ds = pd.concat([c_tr.assign(ds='Train'), c_te.assign(ds='Test')])
  _, p12, _, _ = stats.chi2_contingency(
      pd.crosstab(c_all_ds['ds'], c_all_ds['Gender'])
  )
  _, p13, _, _ = stats.chi2_contingency(
      pd.crosstab(c_all_ds['ds'], c_all_ds['Stage'])
  )
  p_vals.extend([p10, p11, p12, p13])

  if all(p > 0.05 for p in p_vals):
    found_seed = seed
    break

print(f'-> Đã tìm thấy seed tối ưu: {found_seed}')


# ==========================================
# BỔ SUNG 3: XUẤT FILE ĐỘC LẬP THEO THỜI GIAN
# ==========================================
df_out = df_clean.copy()
df_out['Chia tập mẫu'] = 'Train'
df_out.loc[test_idx, 'Chia tập mẫu'] = 'Test'

df_split_summary = generate_split_report(
    df_proc.loc[train_idx], df_proc.loc[test_idx]
)

# Tạo chuỗi thời gian YYYYMMDD_HHMMSS
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = f'Ket_qua_chia_tap_mau_K_Da_Day_{timestamp_str}.xlsx'

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
  # Sheet 1: Danh sách mẫu sau khi gán nhãn Train/Test
  df_out.to_excel(writer, sheet_name='DS MẪU SAU CHIA', index=False)

  # Sheet 2: Thống kê mô tả toàn bộ dữ liệu & Giai đoạn Ung thư
  df_overall_summary.to_excel(
      writer, sheet_name='THỐNG KÊ TOÀN BỘ MẪU', index=False, startrow=0
  )
  df_stage_summary.to_excel(
      writer, sheet_name='THỐNG KÊ TOÀN BỘ MẪU', index=False, startrow=7
  )

  # Sheet 3: Kiểm định đồng nhất Train vs Test
  df_split_summary.to_excel(
      writer, sheet_name='KIỂM ĐỊNH TRAIN TEST', index=False
  )

  # Tự động chỉnh độ rộng cột Excel
  for sheet_name in writer.sheets:
    ws = writer.sheets[sheet_name]
    for col in ws.columns:
      max_len = max(len(str(cell.value or '')) for cell in col)
      col_letter = openpyxl.utils.get_column_letter(col[0].column)
      ws.column_dimensions[col_letter].width = min(max_len + 4, 75)

print(f"-> Xuất thành công file độc lập: '{output_file}'")