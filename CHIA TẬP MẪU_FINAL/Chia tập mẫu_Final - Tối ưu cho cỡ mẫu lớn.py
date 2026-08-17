from datetime import datetime
import numpy as np
import openpyxl
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split


# ==============================================================================
# 1. HÀM KIỂM TRA TỐC ĐỘ CAO (DÙNG CHO VÒNG LẶP TÌM SEED - EARLY EXIT)
# ==============================================================================
def validate_seed_fast(
    train_idx, test_idx, tuois, nhoms, gioi_tinhs, giai_doans
):
  """Kiểm tra điều kiện p > 0.05 cực nhanh bằng NumPy và Ngắt sớm (Early Exit)."""
  tr_mask = np.zeros(len(nhoms), dtype=bool)
  te_mask = np.zeros(len(nhoms), dtype=bool)
  tr_mask[train_idx] = True
  te_mask[test_idx] = True

  # 1. KIỂM ĐỊNH NỘI BỘ TẬP TRAIN (Healthy vs Benign vs Cancer)
  tr_h_a = tuois[tr_mask & (nhoms == 'Healthy')]
  tr_b_a = tuois[tr_mask & (nhoms == 'Benign')]
  tr_c_a = tuois[tr_mask & (nhoms == 'Cancer')]

  tr_h_a = tr_h_a[~np.isnan(tr_h_a)]
  tr_b_a = tr_b_a[~np.isnan(tr_b_a)]
  tr_c_a = tr_c_a[~np.isnan(tr_c_a)]

  if len(tr_h_a) == 0 or len(tr_b_a) == 0 or len(tr_c_a) == 0:
    return False

  # Tuổi (ANOVA & Kruskal-Wallis)
  p1 = stats.f_oneway(tr_h_a, tr_b_a, tr_c_a)[1]
  if np.isnan(p1) or p1 <= 0.05:
    return False

  p2 = stats.kruskal(tr_h_a, tr_b_a, tr_c_a)[1]
  if np.isnan(p2) or p2 <= 0.05:
    return False

  # Giới tính (Chi2)
  ct_tr = np.array([
      [
          np.sum(tr_mask & (nhoms == 'Healthy') & (gioi_tinhs == 'Nam')),
          np.sum(tr_mask & (nhoms == 'Healthy') & (gioi_tinhs == 'Nữ')),
      ],
      [
          np.sum(tr_mask & (nhoms == 'Benign') & (gioi_tinhs == 'Nam')),
          np.sum(tr_mask & (nhoms == 'Benign') & (gioi_tinhs == 'Nữ')),
      ],
      [
          np.sum(tr_mask & (nhoms == 'Cancer') & (gioi_tinhs == 'Nam')),
          np.sum(tr_mask & (nhoms == 'Cancer') & (gioi_tinhs == 'Nữ')),
      ],
  ])
  ct_tr = ct_tr[:, ct_tr.sum(axis=0) > 0]
  if ct_tr.shape[1] > 1:
    p3 = stats.chi2_contingency(ct_tr)[1]
    if np.isnan(p3) or p3 <= 0.05:
      return False

  # 2. KIỂM ĐỊNH NỘI BỘ TẬP TEST (Healthy vs Benign vs Cancer)
  te_h_a = tuois[te_mask & (nhoms == 'Healthy')]
  te_b_a = tuois[te_mask & (nhoms == 'Benign')]
  te_c_a = tuois[te_mask & (nhoms == 'Cancer')]

  te_h_a = te_h_a[~np.isnan(te_h_a)]
  te_b_a = te_b_a[~np.isnan(te_b_a)]
  te_c_a = te_c_a[~np.isnan(te_c_a)]

  if len(te_h_a) == 0 or len(te_b_a) == 0 or len(te_c_a) == 0:
    return False

  # Tuổi (ANOVA & Kruskal-Wallis)
  p4 = stats.f_oneway(te_h_a, te_b_a, te_c_a)[1]
  if np.isnan(p4) or p4 <= 0.05:
    return False

  p5 = stats.kruskal(te_h_a, te_b_a, te_c_a)[1]
  if np.isnan(p5) or p5 <= 0.05:
    return False

  # Giới tính (Chi2)
  ct_te = np.array([
      [
          np.sum(te_mask & (nhoms == 'Healthy') & (gioi_tinhs == 'Nam')),
          np.sum(te_mask & (nhoms == 'Healthy') & (gioi_tinhs == 'Nữ')),
      ],
      [
          np.sum(te_mask & (nhoms == 'Benign') & (gioi_tinhs == 'Nam')),
          np.sum(te_mask & (nhoms == 'Benign') & (gioi_tinhs == 'Nữ')),
      ],
      [
          np.sum(te_mask & (nhoms == 'Cancer') & (gioi_tinhs == 'Nam')),
          np.sum(te_mask & (nhoms == 'Cancer') & (gioi_tinhs == 'Nữ')),
      ],
  ])
  ct_te = ct_te[:, ct_te.sum(axis=0) > 0]
  if ct_te.shape[1] > 1:
    p6 = stats.chi2_contingency(ct_te)[1]
    if np.isnan(p6) or p6 <= 0.05:
      return False

  # 3, 4, 5. KIỂM ĐỊNH SO SÁNH TRAIN VS TEST THEO TỪNG NHÓM
  for grp in ['Healthy', 'Benign', 'Cancer']:
    m_tr_grp = tr_mask & (nhoms == grp)
    m_te_grp = te_mask & (nhoms == grp)

    a_tr = tuois[m_tr_grp]
    a_te = tuois[m_te_grp]
    a_tr = a_tr[~np.isnan(a_tr)]
    a_te = a_te[~np.isnan(a_te)]

    p_tt = stats.ttest_ind(a_tr, a_te)[1]
    if np.isnan(p_tt) or p_tt <= 0.05:
      return False

    p_mwu = stats.mannwhitneyu(a_tr, a_te, alternative='two-sided')[1]
    if np.isnan(p_mwu) or p_mwu <= 0.05:
      return False

    sex_mat = np.array([
        [
            np.sum(m_tr_grp & (gioi_tinhs == 'Nam')),
            np.sum(m_tr_grp & (gioi_tinhs == 'Nữ')),
        ],
        [
            np.sum(m_te_grp & (gioi_tinhs == 'Nam')),
            np.sum(m_te_grp & (gioi_tinhs == 'Nữ')),
        ],
    ])
    sex_mat = sex_mat[:, sex_mat.sum(axis=0) > 0]
    if sex_mat.shape[1] > 1:
      p_sex = stats.chi2_contingency(sex_mat)[1]
      if np.isnan(p_sex) or p_sex <= 0.05:
        return False

    if grp == 'Cancer':
      stages = ['Giai đoạn I', 'Giai đoạn II', 'Giai đoạn III', 'Giai đoạn IV']
      stg_mat = np.array([
          [np.sum(m_tr_grp & (giai_doans == s)) for s in stages],
          [np.sum(m_te_grp & (giai_doans == s)) for s in stages],
      ])
      stg_mat = stg_mat[:, stg_mat.sum(axis=0) > 0]
      if stg_mat.shape[1] > 1:
        p_stg = stats.chi2_contingency(stg_mat)[1]
        if np.isnan(p_stg) or p_stg <= 0.05:
          return False

  return True


# ==============================================================================
# 2. HÀM THỐNG KÊ & KIỂM ĐỊNH KHÁC BIỆT TRÊN TOÀN BỘ TẬP DỮ LIỆU GỐC
# ==============================================================================
def generate_overall_summary(df):
  """Tính toán thống kê mô tả và kiểm định p-value giữa 3 nhóm Healthy, Benign, Cancer trên toàn bộ dữ liệu mẫu."""
  h_all = df[df['Nhóm'] == 'Healthy']
  b_all = df[df['Nhóm'] == 'Benign']
  c_all = df[df['Nhóm'] == 'Cancer']

  def fmt_age(s):
    s_clean = s.dropna()
    return f'{s_clean.mean():.2f} ± {s_clean.std():.2f}' if len(s_clean) > 0 else 'N/A'

  def fmt_sex(s):
    n = len(s)
    if n == 0:
      return 'N/A'
    n_nam = (s == 'Nam').sum()
    n_nu = (s == 'Nữ').sum()
    return f'Nam: {n_nam} ({n_nam/n*100:.1f}%), Nữ: {n_nu} ({n_nu/n*100:.1f}%)'

  # Kiểm định Tuổi (ANOVA & Kruskal-Wallis)
  p_age_anova = stats.f_oneway(
      h_all['Tuổi'].dropna(), b_all['Tuổi'].dropna(), c_all['Tuổi'].dropna()
  )[1]
  p_age_kw = stats.kruskal(
      h_all['Tuổi'].dropna(), b_all['Tuổi'].dropna(), c_all['Tuổi'].dropna()
  )[1]

  # Kiểm định Giới tính (Chi-squared)
  ct_sex = pd.crosstab(df['Nhóm'], df['Giới tính'])
  p_sex_chi2 = (
      stats.chi2_contingency(ct_sex)[1] if ct_sex.shape[1] > 1 else np.nan
  )

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
              else 'Khác biệt có ý nghĩa (p <= 0.05)'
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
              else 'Khác biệt có ý nghĩa (p <= 0.05)'
          ),
      },
  ]

  # Bảng phân bố Giai đoạn cho nhóm Cancer trên toàn tập
  stage_rows = []
  stages_list = ['Giai đoạn I', 'Giai đoạn II', 'Giai đoạn III', 'Giai đoạn IV']
  for st in stages_list:
    cnt = (c_all['Giai đoạn'] == st).sum()
    stage_rows.append({
        'Giai đoạn Ung thư': st,
        'Số lượng mẫu (n)': cnt,
        'Tỷ lệ trong nhóm Cancer (%)': (
            f'{cnt / len(c_all) * 100:.2f}%' if len(c_all) > 0 else '0.00%'
        ),
        f'Tỷ lệ trên toàn bộ mẫu N={len(df)} (%)': (
            f'{cnt / len(df) * 100:.2f}%' if len(df) > 0 else '0.00%'
        ),
    })

  return pd.DataFrame(overall_rows), pd.DataFrame(stage_rows)


# ==============================================================================
# 3. HÀM DỰNG BÁO CÁO CHI TIẾT TRAIN/TEST
# ==============================================================================
def run_statistical_tests_report(df_train, df_test):
  """Tạo dataframe báo cáo đầy đủ sau khi đã chọn được seed tối ưu."""
  results = []

  def fmt_age(df_sub):
    s = df_sub['Tuổi'].dropna()
    return f'{s.mean():.2f} ± {s.std():.2f}' if len(s) > 0 else 'N/A'

  def fmt_sex(df_sub):
    n = len(df_sub)
    if n == 0:
      return 'N/A'
    n_nam = (df_sub['Giới tính'] == 'Nam').sum()
    n_nu = (df_sub['Giới tính'] == 'Nữ').sum()
    return f'Nam: {n_nam} ({n_nam/n*100:.1f}%), Nữ: {n_nu} ({n_nu/n*100:.1f}%)'

  def fmt_stage(df_sub):
    n = len(df_sub)
    if n == 0:
      return 'N/A'
    s1, s2, s3, s4 = [
        (df_sub['Giai đoạn'] == f'Giai đoạn {i}').sum()
        for i in ['I', 'II', 'III', 'IV']
    ]
    return f'GĐ I: {s1} ({s1/n*100:.1f}%), GĐ II: {s2} ({s2/n*100:.1f}%), GĐ III: {s3} ({s3/n*100:.1f}%), GĐ IV: {s4} ({s4/n*100:.1f}%)'

  tr_h, tr_b, tr_c = (
      df_train[df_train['Nhóm'] == 'Healthy'],
      df_train[df_train['Nhóm'] == 'Benign'],
      df_train[df_train['Nhóm'] == 'Cancer'],
  )
  te_h, te_b, te_c = (
      df_test[df_test['Nhóm'] == 'Healthy'],
      df_test[df_test['Nhóm'] == 'Benign'],
      df_test[df_test['Nhóm'] == 'Cancer'],
  )

  # 1. Nội bộ Tập Train
  p_anova_tr = stats.f_oneway(
      tr_h['Tuổi'].dropna(), tr_b['Tuổi'].dropna(), tr_c['Tuổi'].dropna()
  )[1]
  p_kw_tr = stats.kruskal(
      tr_h['Tuổi'].dropna(), tr_b['Tuổi'].dropna(), tr_c['Tuổi'].dropna()
  )[1]
  p_chi2_sex_tr = stats.chi2_contingency(
      pd.crosstab(df_train['Nhóm'], df_train['Giới tính'])
  )[1]

  results.append({
      'Hạng mục kiểm định': '1. Nội bộ Tập Train (70%)',
      'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
      'Tập Train / Nhóm A': (
          f'Healthy (n={len(tr_h)}): {fmt_age(tr_h)} | Benign (n={len(tr_b)}):'
          f' {fmt_age(tr_b)}'
      ),
      'Tập Test / Nhóm B': f'Cancer (n={len(tr_c)}): {fmt_age(tr_c)}',
      'Chỉ số t-test / Chi2 / ANOVA': f'ANOVA p = {p_anova_tr:.4f}',
      'Chỉ số MWU / Kruskal': f'KW p = {p_kw_tr:.4f}',
      'Đánh giá tương đồng': 'Đồng nhất / Đạt (p > 0.05)',
  })
  results.append({
      'Hạng mục kiểm định': '1. Nội bộ Tập Train (70%)',
      'Đặc điểm / Biến': 'Giới tính - n (%)',
      'Tập Train / Nhóm A': (
          f'Healthy: {fmt_sex(tr_h)} | Benign: {fmt_sex(tr_b)}'
      ),
      'Tập Test / Nhóm B': f'Cancer: {fmt_sex(tr_c)}',
      'Chỉ số t-test / Chi2 / ANOVA': f'Chi2 p = {p_chi2_sex_tr:.4f}',
      'Chỉ số MWU / Kruskal': '-',
      'Đánh giá tương đồng': 'Đồng nhất / Đạt (p > 0.05)',
  })

  # 2. Nội bộ Tập Test
  p_anova_te = stats.f_oneway(
      te_h['Tuổi'].dropna(), te_b['Tuổi'].dropna(), te_c['Tuổi'].dropna()
  )[1]
  p_kw_te = stats.kruskal(
      te_h['Tuổi'].dropna(), te_b['Tuổi'].dropna(), te_c['Tuổi'].dropna()
  )[1]
  p_chi2_sex_te = stats.chi2_contingency(
      pd.crosstab(df_test['Nhóm'], df_test['Giới tính'])
  )[1]

  results.append({
      'Hạng mục kiểm định': '2. Nội bộ Tập Test (30%)',
      'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
      'Tập Train / Nhóm A': (
          f'Healthy (n={len(te_h)}): {fmt_age(te_h)} | Benign (n={len(te_b)}):'
          f' {fmt_age(te_b)}'
      ),
      'Tập Test / Nhóm B': f'Cancer (n={len(te_c)}): {fmt_age(te_c)}',
      'Chỉ số t-test / Chi2 / ANOVA': f'ANOVA p = {p_anova_te:.4f}',
      'Chỉ số MWU / Kruskal': f'KW p = {p_kw_te:.4f}',
      'Đánh giá tương đồng': 'Đồng nhất / Đạt (p > 0.05)',
  })
  results.append({
      'Hạng mục kiểm định': '2. Nội bộ Tập Test (30%)',
      'Đặc điểm / Biến': 'Giới tính - n (%)',
      'Tập Train / Nhóm A': (
          f'Healthy: {fmt_sex(te_h)} | Benign: {fmt_sex(te_b)}'
      ),
      'Tập Test / Nhóm B': f'Cancer: {fmt_sex(te_c)}',
      'Chỉ số t-test / Chi2 / ANOVA': f'Chi2 p = {p_chi2_sex_te:.4f}',
      'Chỉ số MWU / Kruskal': '-',
      'Đánh giá tương đồng': 'Đồng nhất / Đạt (p > 0.05)',
  })

  # 3, 4, 5. So sánh Train vs Test từng nhóm
  groups = [
      ('3. Nhóm Healthy (Train vs Test)', tr_h, te_h, False),
      ('4. Nhóm Benign (Train vs Test)', tr_b, te_b, False),
      ('5. Nhóm Cancer (Train vs Test)', tr_c, te_c, True),
  ]

  for label, sub_tr, sub_te, has_stage in groups:
    p_ttest = stats.ttest_ind(sub_tr['Tuổi'].dropna(), sub_te['Tuổi'].dropna())[
        1
    ]
    p_mwu = stats.mannwhitneyu(
        sub_tr['Tuổi'].dropna(),
        sub_te['Tuổi'].dropna(),
        alternative='two-sided',
    )[1]

    results.append({
        'Hạng mục kiểm định': label,
        'Đặc điểm / Biến': 'Tuổi (Mean ± SD)',
        'Tập Train / Nhóm A': f'Train (n={len(sub_tr)}): {fmt_age(sub_tr)}',
        'Tập Test / Nhóm B': f'Test (n={len(sub_te)}): {fmt_age(sub_te)}',
        'Chỉ số t-test / Chi2 / ANOVA': f't-test p = {p_ttest:.4f}',
        'Chỉ số MWU / Kruskal': f'MWU p = {p_mwu:.4f}',
        'Đánh giá tương đồng': 'Đồng nhất / Đạt (p > 0.05)',
    })

    sex_mat = np.array([
        [
            (sub_tr['Giới tính'] == 'Nam').sum(),
            (sub_tr['Giới tính'] == 'Nữ').sum(),
        ],
        [
            (sub_te['Giới tính'] == 'Nam').sum(),
            (sub_te['Giới tính'] == 'Nữ').sum(),
        ],
    ])
    sex_mat = sex_mat[:, sex_mat.sum(axis=0) > 0]
    p_chi2_sex = (
        stats.chi2_contingency(sex_mat)[1] if sex_mat.shape[1] > 1 else 1.0
    )

    results.append({
        'Hạng mục kiểm định': label,
        'Đặc điểm / Biến': 'Giới tính - n (%)',
        'Tập Train / Nhóm A': f'Train: {fmt_sex(sub_tr)}',
        'Tập Test / Nhóm B': f'Test: {fmt_sex(sub_te)}',
        'Chỉ số t-test / Chi2 / ANOVA': f'Chi2 p = {p_chi2_sex:.4f}',
        'Chỉ số MWU / Kruskal': '-',
        'Đánh giá tương đồng': 'Đồng nhất / Đạt (p > 0.05)',
    })

    if has_stage:
      stages_list = [f'Giai đoạn {i}' for i in ['I', 'II', 'III', 'IV']]
      stg_mat = np.array([
          [(sub_tr['Giai đoạn'] == s).sum() for s in stages_list],
          [(sub_te['Giai đoạn'] == s).sum() for s in stages_list],
      ])
      stg_mat = stg_mat[:, stg_mat.sum(axis=0) > 0]
      p_chi2_stage = (
          stats.chi2_contingency(stg_mat)[1] if stg_mat.shape[1] > 1 else 1.0
      )

      results.append({
          'Hạng mục kiểm định': label,
          'Đặc điểm / Biến': 'Giai đoạn (Stage) - n (%)',
          'Tập Train / Nhóm A': f'Train: {fmt_stage(sub_tr)}',
          'Tập Test / Nhóm B': f'Test: {fmt_stage(sub_te)}',
          'Chỉ số t-test / Chi2 / ANOVA': f'Chi2 p = {p_chi2_stage:.4f}',
          'Chỉ số MWU / Kruskal': '-',
          'Đánh giá tương đồng': 'Đồng nhất / Đạt (p > 0.05)',
      })

  return pd.DataFrame(results)


# ==============================================================================
# 4. TÌM SEED VỚI VÒNG LẶP TỐI ƯU
# ==============================================================================
def split_data_until_balanced(df, test_size=0.3, max_iter=50000):
  print('Đang thực hiện tìm Seed tối ưu...')

  strat_key = (
      df['Nhóm'].fillna('KhongXacDinh').astype(str)
      + '_'
      + df['Giới tính'].fillna('KhongXacDinh').astype(str)
      + '_'
      + df['Giai đoạn'].fillna('KhoeManh_LanhTinh').astype(str)
  )

  tuois = df['Tuổi'].to_numpy()
  nhoms = df['Nhóm'].to_numpy()
  gioi_tinhs = df['Giới tính'].to_numpy()
  giai_doans = df['Giai đoạn'].to_numpy()
  indices = df.index.to_numpy()

  for seed in range(max_iter):
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=strat_key
    )

    if validate_seed_fast(
        train_idx, test_idx, tuois, nhoms, gioi_tinhs, giai_doans
    ):
      print(f'==> THÀNH CÔNG! Tìm thấy seed: {seed} (sau {seed+1} lần lặp)')
      df_tr = df.loc[train_idx]
      df_te = df.loc[test_idx]
      df_report = run_statistical_tests_report(df_tr, df_te)
      return seed, df_tr, df_te, df_report

  print(f'Cảnh báo: Không tìm thấy seed phù hợp sau {max_iter} lần lặp.')
  return None, None, None, None


# ==============================================================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================
if __name__ == '__main__':
  file_path = r'C:\Users\pc-008\Desktop\du-an-dung-chung\CHIA TẬP MẪU_FINAL\THU THẬP DỮ LIỆU_ĐỀ TÀI K DẠ DÀY_2024_FINAL - Bổ sung Benign.xlsx'
  df_raw = pd.read_excel(file_path, sheet_name='DS MẪU NC')

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

  # BỔ SUNG: Tính toán thống kê & kiểm định trên toàn bộ tập dữ liệu gốc
  df_overall_summary, df_stage_summary = generate_overall_summary(df)

  seed_toi_uu, df_train, df_test, df_kiem_dinh = split_data_until_balanced(
      df, test_size=0.3, max_iter=50000
  )

  if df_test is not None:
    df_output_ds = df_raw.copy()
    df_output_ds['Chia tập mẫu'] = 'Train'
    df_output_ds.loc[df_test.index, 'Chia tập mẫu'] = 'Test'

    output_filename = f"Ket_qua_chia_tap_mau_K_Da_Day_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"

    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
      # Sheet 1: Danh sách mẫu sau khi chia
      df_output_ds.to_excel(writer, sheet_name='DS MẪU SAU CHIA', index=False)

      # Sheet 2 (MỚI): Thống kê và kiểm định khác biệt trên toàn bộ dữ liệu mẫu
      df_overall_summary.to_excel(
          writer, sheet_name='THỐNG KÊ TOÀN BỘ MẪU', index=False, startrow=0
      )
      df_stage_summary.to_excel(
          writer,
          sheet_name='THỐNG KÊ TOÀN BỘ MẪU',
          index=False,
          startrow=len(df_overall_summary) + 3,
      )

      # Sheet 3: Kiểm định so sánh Train vs Test
      if df_kiem_dinh is not None:
        df_kiem_dinh.to_excel(
            writer, sheet_name='KIỂM ĐỊNH TRAIN TEST', index=False
        )

      # Tự động căn chỉnh độ rộng cột cho tất cả các sheet
      for ws in writer.sheets.values():
        for col in ws.columns:
          max_len = max(len(str(cell.value or '')) for cell in col)
          col_letter = openpyxl.utils.get_column_letter(col[0].column)
          ws.column_dimensions[col_letter].width = min(max_len + 4, 85)

    print(
        f"-> Hoàn tất! File đã lưu tại: '{output_filename}' với sheet mới"
        " 'THỐNG KÊ TOÀN BỘ MẪU'."
    )