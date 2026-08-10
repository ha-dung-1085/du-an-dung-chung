from datetime import datetime
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


# ==============================================================================
# 1. HÀM KIỂM ĐỊNH THỐNG KÊ (BẮT BUỘC TẤT CẢ P-VALUE > 0.05 ĐỂ ĐẠT ĐỒNG NHẤT)
# ==============================================================================
def run_statistical_tests(df_train, df_test):
  """Thực hiện kiểm định thống kê và thu thập toàn bộ p-value."""
  results = []
  all_p_values = []

  # Format hiển thị
  def fmt_age(df_sub):
    s = df_sub['Tuổi'].dropna()
    if len(s) == 0:
      return 'N/A'
    return f'{s.mean():.2f} ± {s.std():.2f}'

  def fmt_sex(df_sub):
    n_total = len(df_sub)
    if n_total == 0:
      return 'N/A'
    n_nam = (df_sub['Giới tính'] == 'Nam').sum()
    n_nu = (df_sub['Giới tính'] == 'Nữ').sum()
    return f'Nam: {n_nam} ({n_nam/n_total*100:.1f}%), Nữ: {n_nu} ({n_nu/n_total*100:.1f}%)'

  def fmt_stage(df_sub):
    n_total = len(df_sub)
    if n_total == 0:
      return 'N/A'
    s1 = (df_sub['Giai đoạn'] == 'Giai đoạn I').sum()
    s2 = (df_sub['Giai đoạn'] == 'Giai đoạn II').sum()
    s3 = (df_sub['Giai đoạn'] == 'Giai đoạn III').sum()
    s4 = (df_sub['Giai đoạn'] == 'Giai đoạn IV').sum()
    return f'GĐ I: {s1} ({s1/n_total*100:.1f}%), GĐ II: {s2} ({s2/n_total*100:.1f}%), GĐ III: {s3} ({s3/n_total*100:.1f}%), GĐ IV: {s4} ({s4/n_total*100:.1f}%)'

  # -------------------------------------------------------------
  # 1. NỘI BỘ TẬP TRAIN (So sánh 3 nhóm: Healthy, Benign, Cancer)
  # -------------------------------------------------------------
  tr_h = df_train[df_train['Nhóm'] == 'Healthy']
  tr_b = df_train[df_train['Nhóm'] == 'Benign']
  tr_c = df_train[df_train['Nhóm'] == 'Cancer']

  # Tuổi - Train (ANOVA & Kruskal-Wallis)
  _, p_anova_tr = stats.f_oneway(
      tr_h['Tuổi'].dropna(), tr_b['Tuổi'].dropna(), tr_c['Tuổi'].dropna()
  )
  _, p_kw_tr = stats.kruskal(
      tr_h['Tuổi'].dropna(), tr_b['Tuổi'].dropna(), tr_c['Tuổi'].dropna()
  )
  all_p_values.extend([p_anova_tr, p_kw_tr])

  results.append({
      'Hạng mục kiểm định': '1. Nội bộ Tập Train (70%)',
      'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
      'Tập A / Nhóm A': f'Healthy (n={len(tr_h)}): {fmt_age(tr_h)}',
      'Tập B / Nhóm B': (
          f'Benign (n={len(tr_b)}): {fmt_age(tr_b)} | Cancer (n={len(tr_c)}):'
          f' {fmt_age(tr_c)}'
      ),
      'Chỉ số t-test / Chi2 / ANOVA': f'ANOVA p = {p_anova_tr:.4f}',
      'Chỉ số MWU / Kruskal': f'KW p = {p_kw_tr:.4f}',
      'Đánh giá tương đồng': (
          'Đồng nhất / Đạt (p > 0.05)'
          if (p_anova_tr > 0.05 and p_kw_tr > 0.05)
          else 'Khác biệt / Chưa đạt (p < 0.05)'
      ),
  })

  # Giới tính - Train (Chi-squared)
  ct_sex_tr = pd.crosstab(df_train['Nhóm'], df_train['Giới tính'])
  _, p_chi2_sex_tr, _, _ = stats.chi2_contingency(ct_sex_tr)
  all_p_values.append(p_chi2_sex_tr)

  results.append({
      'Hạng mục kiểm định': '1. Nội bộ Tập Train (70%)',
      'Đặc điểm / Biến': 'Giới tính - n (%)',
      'Tập A / Nhóm A': f'Healthy: {fmt_sex(tr_h)}',
      'Tập B / Nhóm B': f'Benign: {fmt_sex(tr_b)} | Cancer: {fmt_sex(tr_c)}',
      'Chỉ số t-test / Chi2 / ANOVA': f'Chi2 p = {p_chi2_sex_tr:.4f}',
      'Chỉ số MWU / Kruskal': '-',
      'Đánh giá tương đồng': (
          'Đồng nhất / Đạt (p > 0.05)'
          if p_chi2_sex_tr > 0.05
          else 'Khác biệt / Chưa đạt (p < 0.05)'
      ),
  })

  # -------------------------------------------------------------
  # 2. NỘI BỘ TẬP TEST (So sánh 3 nhóm: Healthy, Benign, Cancer)
  # -------------------------------------------------------------
  te_h = df_test[df_test['Nhóm'] == 'Healthy']
  te_b = df_test[df_test['Nhóm'] == 'Benign']
  te_c = df_test[df_test['Nhóm'] == 'Cancer']

  # Tuổi - Test (ANOVA & Kruskal-Wallis)
  _, p_anova_te = stats.f_oneway(
      te_h['Tuổi'].dropna(), te_b['Tuổi'].dropna(), te_c['Tuổi'].dropna()
  )
  _, p_kw_te = stats.kruskal(
      te_h['Tuổi'].dropna(), te_b['Tuổi'].dropna(), te_c['Tuổi'].dropna()
  )
  all_p_values.extend([p_anova_te, p_kw_te])

  results.append({
      'Hạng mục kiểm định': '2. Nội bộ Tập Test (30%)',
      'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
      'Tập A / Nhóm A': f'Healthy (n={len(te_h)}): {fmt_age(te_h)}',
      'Tập B / Nhóm B': (
          f'Benign (n={len(te_b)}): {fmt_age(te_b)} | Cancer (n={len(te_c)}):'
          f' {fmt_age(te_c)}'
      ),
      'Chỉ số t-test / Chi2 / ANOVA': f'ANOVA p = {p_anova_te:.4f}',
      'Chỉ số MWU / Kruskal': f'KW p = {p_kw_te:.4f}',
      'Đánh giá tương đồng': (
          'Đồng nhất / Đạt (p > 0.05)'
          if (p_anova_te > 0.05 and p_kw_te > 0.05)
          else 'Khác biệt / Chưa đạt (p < 0.05)'
      ),
  })

  # Giới tính - Test (Chi-squared)
  ct_sex_te = pd.crosstab(df_test['Nhóm'], df_test['Giới tính'])
  _, p_chi2_sex_te, _, _ = stats.chi2_contingency(ct_sex_te)
  all_p_values.append(p_chi2_sex_te)

  results.append({
      'Hạng mục kiểm định': '2. Nội bộ Tập Test (30%)',
      'Đặc điểm / Biến': 'Giới tính - n (%)',
      'Tập A / Nhóm A': f'Healthy: {fmt_sex(te_h)}',
      'Tập B / Nhóm B': f'Benign: {fmt_sex(te_b)} | Cancer: {fmt_sex(te_c)}',
      'Chỉ số t-test / Chi2 / ANOVA': f'Chi2 p = {p_chi2_sex_te:.4f}',
      'Chỉ số MWU / Kruskal': '-',
      'Đánh giá tương đồng': (
          'Đồng nhất / Đạt (p > 0.05)'
          if p_chi2_sex_te > 0.05
          else 'Khác biệt / Chưa đạt (p < 0.05)'
      ),
  })

  # -------------------------------------------------------------
  # 3. SO SÁNH TRAIN VS TEST THEO TỪNG NHÓM BỆNH
  # -------------------------------------------------------------
  groups_to_compare = [
      ('3. Nhóm Healthy (Train vs Test)', tr_h, te_h, False),
      ('4. Nhóm Benign (Train vs Test)', tr_b, te_b, False),
      ('5. Nhóm Cancer (Train vs Test)', tr_c, te_c, True),
  ]

  for label, sub_tr, sub_te, has_stage in groups_to_compare:
    # a) Tuổi (t-test & Mann-Whitney U)
    _, p_ttest = stats.ttest_ind(
        sub_tr['Tuổi'].dropna(), sub_te['Tuổi'].dropna()
    )
    _, p_mwu = stats.mannwhitneyu(
        sub_tr['Tuổi'].dropna(),
        sub_te['Tuổi'].dropna(),
        alternative='two-sided',
    )
    all_p_values.extend([p_ttest, p_mwu])

    results.append({
        'Hạng mục kiểm định': label,
        'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
        'Tập A / Nhóm A': f'Train (n={len(sub_tr)}): {fmt_age(sub_tr)}',
        'Tập B / Nhóm B': f'Test (n={len(sub_te)}): {fmt_age(sub_te)}',
        'Chỉ số t-test / Chi2 / ANOVA': f't-test p = {p_ttest:.4f}',
        'Chỉ số MWU / Kruskal': f'MWU p = {p_mwu:.4f}',
        'Đánh giá tương đồng': (
            'Đồng nhất / Đạt (p > 0.05)'
            if (p_ttest > 0.05 and p_mwu > 0.05)
            else 'Khác biệt / Chưa đạt (p < 0.05)'
        ),
    })

    # b) Giới tính (Chi-squared)
    sex_matrix = [
        [
            (sub_tr['Giới tính'] == 'Nam').sum(),
            (sub_tr['Giới tính'] == 'Nữ').sum(),
        ],
        [
            (sub_te['Giới tính'] == 'Nam').sum(),
            (sub_te['Giới tính'] == 'Nữ').sum(),
        ],
    ]
    _, p_chi2_sex, _, _ = stats.chi2_contingency(sex_matrix)
    all_p_values.append(p_chi2_sex)

    results.append({
        'Hạng mục kiểm định': label,
        'Đặc điểm / Biến': 'Giới tính - n (%)',
        'Tập A / Nhóm A': f'Train: {fmt_sex(sub_tr)}',
        'Tập B / Nhóm B': f'Test: {fmt_sex(sub_te)}',
        'Chỉ số t-test / Chi2 / ANOVA': f'Chi2 p = {p_chi2_sex:.4f}',
        'Chỉ số MWU / Kruskal': '-',
        'Đánh giá tương đồng': (
            'Đồng nhất / Đạt (p > 0.05)'
            if p_chi2_sex > 0.05
            else 'Khác biệt / Chưa đạt (p < 0.05)'
        ),
    })

    # c) Giai đoạn (nếu là nhóm Cancer)
    if has_stage:
      stages_list = [
          'Giai đoạn I',
          'Giai đoạn II',
          'Giai đoạn III',
          'Giai đoạn IV',
      ]
      stage_matrix = [
          [(sub_tr['Giai đoạn'] == s).sum() for s in stages_list],
          [(sub_te['Giai đoạn'] == s).sum() for s in stages_list],
      ]
      # Loại bỏ các cột toàn 0 nếu có để tránh lỗi chi2
      stage_matrix = np.array(stage_matrix)
      stage_matrix = stage_matrix[:, stage_matrix.sum(axis=0) > 0]

      if stage_matrix.shape[1] > 1:
        _, p_chi2_stage, _, _ = stats.chi2_contingency(stage_matrix)
      else:
        p_chi2_stage = 1.0

      all_p_values.append(p_chi2_stage)

      results.append({
          'Hạng mục kiểm định': label,
          'Đặc điểm / Biến': 'Giai đoạn (Stage) - n (%)',
          'Tập A / Nhóm A': f'Train: {fmt_stage(sub_tr)}',
          'Tập B / Nhóm B': f'Test: {fmt_stage(sub_te)}',
          'Chỉ số t-test / Chi2 / ANOVA': f'Chi2 p = {p_chi2_stage:.4f}',
          'Chỉ số MWU / Kruskal': '-',
          'Đánh giá tương đồng': (
              'Đồng nhất / Đạt (p > 0.05)'
              if p_chi2_stage > 0.05
              else 'Khác biệt / Chưa đạt (p < 0.05)'
          ),
      })

  return pd.DataFrame(results), all_p_values


# ==============================================================================
# 2. HÀM TÌM SEED THỎA MÃN TẤT CẢ P-VALUE > 0.05
# ==============================================================================
def split_data_until_balanced(df, test_size=0.3, max_iter=50000):
  """Lặp ngẫu nhiên Stratified Split cho đến khi TẤT CẢ phép kiểm định có p > 0.05."""
  print(
      'Đang thực hiện phân chia tập mẫu và tìm Seed để TẤT CẢ phép kiểm định'
      ' ĐẠT p > 0.05...'
  )

  # ĐÃ SỬA LỖI: Điền giá trị rỗng và chuyển ép kiểu về string cho toàn bộ các cột
  strat_key = (
      df['Nhóm'].fillna('KhongXacDinh').astype(str)
      + '_'
      + df['Giới tính'].fillna('KhongXacDinh').astype(str)
      + '_'
      + df['Giai đoạn'].fillna('KhoeManh_LanhTinh').astype(str)
  )

  found = False
  best_seed = None
  df_train_best = None
  df_test_best = None
  df_report_best = None

  for seed in range(max_iter):
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, random_state=seed, stratify=strat_key
    )

    df_tr = df.loc[train_idx]
    df_te = df.loc[test_idx]

    df_report, all_p_vals = run_statistical_tests(df_tr, df_te)

    # ĐIỀU KIỆN QUYẾT ĐỊNH: TẤT CẢ P-VALUE PHẢI > 0.05
    if all(p > 0.05 for p in all_p_vals if not np.isnan(p)):
      found = True
      best_seed = seed
      df_train_best = df_tr
      df_test_best = df_te
      df_report_best = df_report
      print(
          f'==> THÀNH CÔNG! Đã tìm thấy seed tối ưu: random_state = {seed}'
          f' (sau {seed+1} lần lặp)'
      )
      break

  if not found:
    print(
        'Cảnh báo: Không tìm thấy seed thỏa mãn p > 0.05 cho TẤT CẢ tiêu chí'
        f' sau {max_iter} lần lặp.'
    )

  return best_seed, df_train_best, df_test_best, df_report_best


# ==============================================================================
# 3. HÀM MÔ TẢ TOÀN BỘ MẪU
# ==============================================================================
def generate_overall_summary(df):
  """Tạo bảng thống kê mô tả toàn bộ dữ liệu mẫu ban đầu."""
  h_all = df[df['Nhóm'] == 'Healthy']
  b_all = df[df['Nhóm'] == 'Benign']
  c_all = df[df['Nhóm'] == 'Cancer']

  def fmt_age(s):
    s_clean = s.dropna()
    if len(s_clean) == 0:
      return 'N/A'
    return f'{s_clean.mean():.2f} ± {s_clean.std():.2f}'

  def fmt_sex(s):
    n = len(s)
    if n == 0:
      return 'N/A'
    n_nam = (s == 'Nam').sum()
    n_nu = (s == 'Nữ').sum()
    return f'Nam: {n_nam} ({n_nam/n*100:.1f}%), Nữ: {n_nu} ({n_nu/n*100:.1f}%)'

  # Kiểm định Tuổi toàn mẫu
  _, p_age_anova = stats.f_oneway(
      h_all['Tuổi'].dropna(), b_all['Tuổi'].dropna(), c_all['Tuổi'].dropna()
  )
  _, p_age_kw = stats.kruskal(
      h_all['Tuổi'].dropna(), b_all['Tuổi'].dropna(), c_all['Tuổi'].dropna()
  )

  # Kiểm định Giới tính toàn mẫu
  ct_sex = pd.crosstab(df['Nhóm'], df['Giới tính'])
  _, p_sex_chi2, _, _ = stats.chi2_contingency(ct_sex)

  overall_rows = [
      {
          'Đặc điểm / Biến': 'Tổng số mẫu (N)',
          f'Nhóm Healthy (N={len(h_all)})': len(h_all),
          f'Nhóm Benign (N={len(b_all)})': len(b_all),
          f'Nhóm Cancer (N={len(c_all)})': len(c_all),
          f'Toàn bộ mẫu (N={len(df)})': len(df),
          'Phương pháp kiểm định': '-',
          'P-value': '-',
          'Đánh giá / Nhận xét': 'Mẫu dữ liệu gốc chưa phân chia',
      },
      {
          'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
          f'Nhóm Healthy (N={len(h_all)})': fmt_age(h_all['Tuổi']),
          f'Nhóm Benign (N={len(b_all)})': fmt_age(b_all['Tuổi']),
          f'Nhóm Cancer (N={len(c_all)})': fmt_age(c_all['Tuổi']),
          f'Toàn bộ mẫu (N={len(df)})': fmt_age(df['Tuổi']),
          'Phương pháp kiểm định': 'ANOVA / Kruskal-Wallis',
          'P-value': f'ANOVA p = {p_age_anova:.4f} | KW p = {p_age_kw:.4f}',
          'Đánh giá / Nhận xét': (
              'Đồng nhất / Đạt (p > 0.05)'
              if (p_age_anova > 0.05 and p_age_kw > 0.05)
              else 'Khác biệt / Chưa đạt (p < 0.05)'
          ),
      },
      {
          'Đặc điểm / Biến': 'Giới tính - n (%)',
          f'Nhóm Healthy (N={len(h_all)})': fmt_sex(h_all['Giới tính']),
          f'Nhóm Benign (N={len(b_all)})': fmt_sex(b_all['Giới tính']),
          f'Nhóm Cancer (N={len(c_all)})': fmt_sex(c_all['Giới tính']),
          f'Toàn bộ mẫu (N={len(df)})': fmt_sex(df['Giới tính']),
          'Phương pháp kiểm định': 'Chi-squared Test',
          'P-value': f'Chi2 p = {p_sex_chi2:.4f}',
          'Đánh giá / Nhận xét': (
              'Đồng nhất / Đạt (p > 0.05)'
              if p_sex_chi2 > 0.05
              else 'Khác biệt / Chưa đạt (p < 0.05)'
          ),
      },
  ]
  df_overall = pd.DataFrame(overall_rows)

  stage_rows = []
  cancer_stages = sorted(
      [s for s in c_all['Giai đoạn'].unique() if pd.notna(s)]
  )
  for st in cancer_stages:
    cnt = (c_all['Giai đoạn'] == st).sum()
    prop_cancer = cnt / len(c_all) * 100
    prop_total = cnt / len(df) * 100
    stage_rows.append({
        'Giai đoạn Ung thư': f'{st}',
        'Số lượng mẫu (n)': cnt,
        'Tỷ lệ trong nhóm Cancer (%)': f'{prop_cancer:.2f}%',
        f'Tỷ lệ trên toàn bộ mẫu N={len(df)} (%)': f'{prop_total:.2f}%',
    })
  df_stage = pd.DataFrame(stage_rows)

  return df_overall, df_stage


# ==============================================================================
# 4. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
if __name__ == '__main__':
  # 1. Đọc dữ liệu
  file_path = 'THU THẬP DỮ LIỆU_ĐỀ TÀI K DẠ DÀY_2024_FINAL.xlsx'
  df_raw = pd.read_excel(file_path, sheet_name='DS MẪU NC')

  # 2. Chuẩn hóa dữ liệu
  df = df_raw.copy().reset_index(drop=True)
  df['Nhóm'] = df['NHÓM'].astype(str).str.strip()
  df['Tuổi'] = pd.to_numeric(df['Tuổi'], errors='coerce')

  def clean_gender(val):
    if pd.isna(val):
      return 'KhongXacDinh'
    v = str(val).strip().upper()
    if v in ['NAM', 'M']:
      return 'Nam'
    elif v in ['NỮ', 'NU', 'F']:
      return 'Nữ'
    return v

  df['Giới tính'] = df['Giới'].apply(clean_gender)

  def clean_stage(val):
    if pd.isna(val):
      return None
    v = str(val).strip()
    if v in ['1', 'I', 'Giai đoạn I', 'Giai đoạn 1']:
      return 'Giai đoạn I'
    elif v in ['2', 'II', 'Giai đoạn II', 'Giai đoạn 2']:
      return 'Giai đoạn II'
    elif v in ['3', 'III', 'Giai đoạn III', 'Giai đoạn 3']:
      return 'Giai đoạn III'
    elif v in ['4', 'IV', 'Giai đoạn IV', 'Giai đoạn 4']:
      return 'Giai đoạn IV'
    return v

  df['Giai đoạn'] = df['Giai đoạn'].apply(clean_stage)

  # 3. Phân chia tập Train/Test & Tìm Seed sao cho 100% phép kiểm định ĐẠT (p > 0.05)
  seed_toi_uu, df_train, df_test, df_kiem_dinh = split_data_until_balanced(
      df, test_size=0.3, max_iter=50000
  )

  # 4. Tạo thống kê mô tả toàn bộ dữ liệu mẫu
  df_overall_summary, df_stage_summary = generate_overall_summary(df)

  # 5. Gắn nhãn tập mẫu
  df_output_ds = df_raw.copy()
  df_output_ds['Chia tập mẫu'] = 'Train'
  if df_test is not None:
    df_output_ds.loc[df_test.index, 'Chia tập mẫu'] = 'Test'

  # 6. Xuất file Excel
  timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
  output_filename = f'Ket_qua_chia_tap_mau_K_Da_Day_{timestamp_str}.xlsx'

  with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    df_output_ds.to_excel(writer, sheet_name='DS MẪU SAU CHIA', index=False)
    df_overall_summary.to_excel(
        writer, sheet_name='THỐNG KÊ TOÀN BỘ MẪU', index=False, startrow=0
    )
    df_stage_summary.to_excel(
        writer, sheet_name='THỐNG KÊ TOÀN BỘ MẪU', index=False, startrow=7
    )

    if df_kiem_dinh is not None:
      df_kiem_dinh.to_excel(
          writer, sheet_name='KIỂM ĐỊNH TRAIN TEST', index=False
      )

    for sheet_name in writer.sheets:
      ws = writer.sheets[sheet_name]
      for col in ws.columns:
        max_len = max(len(str(cell.value or '')) for cell in col)
        col_letter = openpyxl.utils.get_column_letter(col[0].column)
        ws.column_dimensions[col_letter].width = min(max_len + 4, 80)

  print(f"-> Hoàn tất! File kết quả đã lưu tại: '{output_filename}'")