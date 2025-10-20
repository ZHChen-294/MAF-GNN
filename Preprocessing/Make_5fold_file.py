import os
import pandas as pd
from sklearn.model_selection import KFold

# =========================================================
# 1. Path setup
# =========================================================
folder_path = ''
save_dir = ''
os.makedirs(save_dir, exist_ok=True)

file_names = os.listdir(folder_path)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# =========================================================
# 2. Generate test sets (one CSV per fold)
# =========================================================
for fold, (_, test_idx) in enumerate(kf.split(file_names), 1):
    test_csv = os.path.join(save_dir, f'MDD_test_data_list_{fold}.csv')
    if os.path.exists(test_csv):
        print(f"[Skip] Test set {fold} already exists.")
        continue

    df_test = pd.DataFrame([file_names[i] for i in test_idx], columns=['Subject ID'])
    df_test.to_csv(test_csv, index=False)
    print(f"[Saved] Test set {fold} with {len(df_test)} subjects.")

# =========================================================
# 3. Generate corresponding train sets
# =========================================================
all_ids = set(file_names)
test_files = [os.path.join(save_dir, f'MDD_test_data_list_{i}.csv') for i in range(1, 6)]

for i, path in enumerate(test_files, 1):
    train_csv = os.path.join(save_dir, f'MDD_train_data_list_{i}.csv')
    if os.path.exists(train_csv):
        print(f"[Skip] Train set {i} already exists.")
        continue

    test_df = pd.read_csv(path)
    test_ids = set(test_df['Subject ID'].astype(str))
    train_ids = all_ids - test_ids

    pd.DataFrame(sorted(train_ids), columns=['Subject ID']).to_csv(train_csv, index=False)
    print(f"[Saved] Train set {i} with {len(train_ids)} subjects.")
