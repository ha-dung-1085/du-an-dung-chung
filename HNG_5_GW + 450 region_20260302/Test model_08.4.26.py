import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict # Thêm cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# Cấu hình thẩm mỹ cho biểu đồ
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100

# =================================================================
# BƯỚC 1: NHẬP DỮ LIỆU & MÔ TẢ THỐNG KÊ
# =================================================================
tmd_path = r"C:\Users\XN\du-an-dung-chung\HNG_5_GW + 450 region_20260302\450 Region\TMD.csv"
info_path = r"C:\Users\XN\du-an-dung-chung\HNG_5_GW + 450 region_20260302\450 Region\HNG.xlsx"

tmd_df = pd.read_csv(tmd_path)
info_df = pd.read_excel(info_path, sheet_name='QC V1')
info_df.columns = info_df.columns.str.strip()

data = pd.merge(info_df, tmd_df, on='SampleID')

print("--- [BƯỚC 1: MÔ TẢ TẬP MẪU] ---")
print(data.groupby(['Cohort', 'Label']).size().unstack(fill_value=0))
print(f"\nCấu trúc: {data.shape[0]} mẫu, {tmd_df.shape[1]-1} đặc trưng TMD.")

# =================================================================
# BƯỚC 2: TÁCH DỮ LIỆU & TIỀN XỬ LÝ
# =================================================================
train_df = data[data['Cohort'] == 'Discovery'].copy()
test_df = data[data['Cohort'] == 'Validation'].copy()

features = [c for c in tmd_df.columns if c != 'SampleID']

X_train_raw = train_df[features].select_dtypes(include=[np.number])
y_train = train_df['Label']

X_test_raw = test_df[features].select_dtypes(include=[np.number])
y_test = test_df['Label']

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train_raw))
X_test_scaled = scaler.transform(imputer.transform(X_test_raw))

# =================================================================
# BƯỚC 3: KHẢO SÁT PCA & LDA
# =================================================================
print("\n--- [BƯỚC 2: KHẢO SÁT PCA & LDA] ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

lda = LDA(n_components=1)
X_lda = lda.fit_transform(X_train_scaled, y_train)

fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_train, palette='Set1', ax=ax[0])
ax[0].set_title('PCA (Discovery Set)')

sns.histplot(x=X_lda.ravel(), hue=y_train, kde=True, ax=ax[1])
ax[1].set_title('LDA Projection (Discovery Set)')
plt.show()

# =================================================================
# BƯỚC 4: TUNING (10-FOLD CV)
# =================================================================
print("\n--- [BƯỚC 3: TUNING MODEL] ---")
param_grid = {'C': np.logspace(-4, 1, 15), 'penalty': ['l1'], 'solver': ['liblinear']}
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    LogisticRegression(class_weight='balanced', max_iter=2000),
    param_grid, cv=cv_strategy, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# =================================================================
# BƯỚC 5: HIỂN THỊ KẾT QUẢ AUC (SỬ DỤNG CROSS-VALIDATION CHO TRAIN)
# =================================================================

# THAY ĐỔI Ở ĐÂY: Thay vì predict trực tiếp trên toàn bộ X_train_scaled,
# ta dùng cross_val_predict để lấy xác suất "khách quan" hơn cho tập Discovery.
y_prob_train_cv = cross_val_predict(
    best_model, X_train_scaled, y_train, 
    cv=cv_strategy, method='predict_proba'
)[:, 1]

auc_train_cv = roc_auc_score(y_train, y_prob_train_cv)
fpr_train, tpr_train, _ = roc_curve(y_train, y_prob_train_cv)

# Dự báo trên tập Validation (giữ nguyên vì đây là dữ liệu hoàn toàn mới)
y_prob_test = best_model.predict_proba(X_test_scaled)[:, 1]
auc_test = roc_auc_score(y_test, y_prob_test)
fpr_test, tpr_test, _ = roc_curve(y_test, y_prob_test)

print("\n" + "="*50)
print(f"KẾT QUẢ AUC (CV-Adjusted):")
print(f"- Discovery (10-Fold CV) AUC: {auc_train_cv:.4f}")
print(f"- Validation AUC:             {auc_test:.4f}")
print(f"- Số đặc trưng được chọn (Lasso): {np.sum(best_model.coef_ != 0)}")
print("="*50)

# Vẽ ROC so sánh
plt.figure(figsize=(7, 6))
plt.plot(fpr_train, tpr_train, label=f'Discovery (CV AUC = {auc_train_cv:.3f})', color='blue')
plt.plot(fpr_test, tpr_test, label=f'Validation (AUC = {auc_test:.3f})', color='red')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('ROC Curve - Cross Validated Train vs Independent Test')
plt.legend()
plt.show()

# Confusion Matrix trên tập Validation
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, best_model.predict(X_test_scaled)), annot=True, fmt='d', cmap='YlGnBu')
plt.title('Confusion Matrix (Validation Set)')
plt.show()