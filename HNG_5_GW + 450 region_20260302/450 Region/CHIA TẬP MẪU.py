import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency
import os
import shutil

# --- CẤU HÌNH ĐƯỜNG DẪN ---
FILE_PATH = r"C:\Users\XN\du-an-dung-chung\HNG_5_GW + 450 region_20260302\450 Region\HNG.xlsx"

def check_similarity(g1, g2, is_gastric_only=False):
    # Kiểm định Age
    _, p_age = ttest_ind(g1['Age'].dropna(), g2['Age'].dropna(), nan_policy='omit')
    
    # Kiểm định Gender
    con_gender = pd.crosstab(pd.concat([g1['Gender'], g2['Gender']]), 
                             ['G1']*len(g1) + ['G2']*len(g2))
    p_gender = chi2_contingency(con_gender)[1] if con_gender.size >= 4 else 1.0
    
    # Kiểm định Stage short
    p_stage = 1.0
    if is_gastric_only:
        stages_of_interest = ['I', 'II', 'IIIA', 'Zero']
        s1 = g1[g1['Stage short'].isin(stages_of_interest)]['Stage short']
        s2 = g2[g2['Stage short'].isin(stages_of_interest)]['Stage short']
        con_stage = pd.crosstab(pd.concat([s1, s2]), ['G1']*len(s1) + ['G2']*len(s2))
        p_stage = chi2_contingency(con_stage)[1] if con_stage.size >= 4 else 1.0
            
    return p_age, p_gender, p_stage

def run_research_sampling():
    print("\n" + "="*50)
    print(" BẮT ĐẦU KIỂM TRA VÀ CHIA MẪU ")
    print("="*50)

    try:
        # 1. Kiểm tra file và Load dữ liệu
        if not os.path.exists(FILE_PATH):
            print(f"X LỖI: Không tìm thấy file tại {FILE_PATH}")
            return

        all_sheets = pd.read_excel(FILE_PATH, sheet_name=None)
        df = all_sheets['QC V1']
        
        # Tiền xử lý dữ liệu: Ép kiểu và xóa Cohort cũ
        df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
        df['Cohort'] = np.nan 

        # Kiểm tra số lượng mẫu
        df_g = df[df['Type'] == 'Gastric'].copy()
        df_h = df[df['Type'] == 'Healthy'].copy()
        
        print(f"[*] Dữ liệu hiện có: Gastric={len(df_g)}, Healthy={len(df_h)}")
        if len(df_g) < 110 or len(df_h) < 110:
            print("X LỖI: Không đủ 110 mẫu mỗi loại để chia.")
            return

        # 2. Vòng lặp tìm mẫu
        found = False
        threshold = 0.05 # Ngưỡng p-value mục tiêu
        max_attempts = 10000 
        
        print(f"[*] Đang tìm tổ hợp mẫu (Target p > {threshold})...")
        
        for i in range(max_attempts):
            # Lấy mẫu ngẫu nhiên
            s_g = df_g.sample(n=110)
            s_h = df_h.sample(n=110)
            
            # Kiểm tra các điều kiện
            p_age_g, p_gen_g, p_stage_g = check_similarity(s_g, df_g[~df_g.index.isin(s_g.index)], True)
            p_age_gh, p_gen_gh, _ = check_similarity(s_g, s_h, False)
            
            # Điều kiện dừng
            if all(p > threshold for p in [p_age_g, p_gen_g, p_stage_g, p_age_gh, p_gen_gh]):
                df.loc[pd.concat([s_g, s_h]).index, 'Cohort'] = 'Discovery'
                found = True
                print(f"✔ THÀNH CÔNG tại lượt thử thứ {i+1}!")
                break
            
            # Cơ chế nới lỏng nếu quá khó (sau 8000 lần thử)
            if i == 8000:
                threshold = 0.02
                print("[!] Cảnh báo: Điều kiện quá khó, đang hạ ngưỡng p-value xuống 0.02...")

        if not found:
            print("X THẤT BẠI: Không tìm được bộ mẫu thỏa mãn sau 10000 lần thử.")
            return

        # 3. In kết quả trực tiếp ra Terminal
        print("\n" + "-"*50)
        print(f"{'CHỈ SỐ':<30} | {'P-VALUE':<10}")
        print("-"*50)
        print(f"{'Gastric Stage Similarity':<30} | {p_stage_g:.4f}")
        print(f"{'Gastric Age Similarity':<30} | {p_age_g:.4f}")
        print(f"{'Gastric vs Healthy Age':<30} | {p_age_gh:.4f}")
        print(f"{'Gastric vs Healthy Gender':<30} | {p_gen_gh:.4f}")
        print("-"*50)

        # 4. Lưu file và Backup
        backup_path = FILE_PATH.replace(".xlsx", "_BACKUP.xlsx")
        shutil.copy(FILE_PATH, backup_path)
        
        all_sheets['QC V1'] = df
        with pd.ExcelWriter(FILE_PATH, engine='openpyxl') as writer:
            for name, sheet in all_sheets.items():
                sheet.to_excel(writer, sheet_name=name, index=False)
        
        print(f"✔ ĐÃ LƯU FILE GỐC VÀ BACKUP THÀNH CÔNG.")
        
        # Hiện biểu đồ kiểm chứng
        plt.figure(figsize=(10,4))
        sns.boxplot(data=df[df['Type'].isin(['Gastric','Healthy'])].assign(G=df['Cohort'].fillna('Remained')), x='Type', y='Age', hue='G')
        plt.title("Verify Age Distribution")
        plt.show()

    except PermissionError:
        print("X LỖI: Hãy ĐÓNG file Excel trước khi chạy code!")
    except Exception as e:
        print(f"X LỖI KHÔNG XÁC ĐỊNH: {str(e)}")

if __name__ == "__main__":
    run_research_sampling()