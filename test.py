from scipy.stats import wilcoxon

# Tạo dữ liệu mẫu: điểm số trước và sau khi can thiệp
truoc = [68, 75, 80, 71, 90, 74, 85, 69, 77, 73]
sau   = [70, 78, 82, 72, 91, 76, 87, 70, 79, 74]

# Thực hiện kiểm định Wilcoxon
stat, p = wilcoxon(truoc, sau)

print("Giá trị thống kê:", stat)
print("Giá trị p:", p)

# Đánh giá kết quả
alpha = 0.05
if p < alpha:
    print("→ Có sự khác biệt có ý nghĩa thống kê (bác bỏ H0).")
else:
    print("→ Không có sự khác biệt đáng kể (không bác bỏ H0).")
