import pandas as pd
import os

# Đường dẫn file (sử dụng r để tránh lỗi dấu gạch chéo)
file_path = r"C:\Users\XN\du-an-dung-chung\HNG_metadata_20260320_Hồ bàn giao.xlsx"

if os.path.exists(file_path):
    try:
        # Đọc tất cả các sheet
        all_sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
        
        print(f"=== BÁO CÁO CHI TIẾT FILE EXCEL ===")
        print(f"Số lượng sheet tìm thấy: {len(all_sheets)}")
        print("=" * 60)

        for sheet_name, df in all_sheets.items():
            print(f"\n📄 SHEET: [{sheet_name}]")
            
            if df.empty:
                print("   ⚠️ Sheet này không có dữ liệu.")
                continue

            # 1. Tổng số lượng dòng và cột
            num_rows, num_cols = df.shape
            print(f"   1. Quy mô: {num_rows} dòng x {num_cols} cột")

            # 2. Mô tả dữ liệu 5 dòng đầu và 5 dòng cuối
            print(f"   2. Kiểm tra các dòng dữ liệu:")
            if num_rows <= 10:
                print("      (Sheet có ít hơn 10 dòng, hiển thị toàn bộ)")
                print(df)
            else:
                print("      --- 5 dòng đầu tiên ---")
                print(df.head(5))
                print("\n      --- 5 dòng cuối cùng ---")
                print(df.tail(5))

            # 3. Mô tả dữ liệu 5 cột đầu và 5 cột cuối
            print(f"\n   3. Kiểm tra các cột dữ liệu:")
            if num_cols <= 10:
                print(f"      Dữ liệu tất cả các cột ({num_cols} cột):")
                print(df.iloc[:, :])
            else:
                # Lấy 5 cột đầu tiên
                head_cols = df.iloc[:, :5]
                # Lấy 5 cột cuối cùng
                tail_cols = df.iloc[:, -5:]
                
                print("      --- 5 cột đầu tiên ---")
                print(head_cols.head(3)) # Hiển thị 3 dòng mẫu của 5 cột này
                
                print("\n      --- 5 cột cuối cùng ---")
                print(tail_cols.head(3)) # Hiển thị 3 dòng mẫu của 5 cột này
            
            print("-" * 60)

    except Exception as e:
        print(f"❌ Có lỗi xảy ra: {e}")
else:
    print("❌ Lỗi: Không tìm thấy file. Vui lòng kiểm tra lại đường dẫn.")