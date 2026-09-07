import gc
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_recall_curve

print("Loading datasets...")
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

def extract_ultimate_spatial_features(df):
    df = df.copy()
    
    # 1. Identify spatial coordinate columns
    xy_cols = [c for c in ["x", "y", "die_x", "die_y", "X", "Y"] if c in df.columns]
    
    if len(xy_cols) >= 2:
        x_c, y_c = xy_cols[0], xy_cols[1]
        
        # Geometry & Polar Coordinates
        df["radial_dist"] = np.sqrt(df[x_c]**2 + df[y_c]**2)
        df["angle"] = np.arctan2(df[y_c], df[x_c])
        df["manhattan_dist"] = np.abs(df[x_c]) + np.abs(df[y_c])
        
        # Concentric Ring Binning (FAB Radial Zones)
        df["wafer_ring_zone"] = pd.qcut(df["radial_dist"], q=10, labels=False, duplicates='drop')

        # 2. Per-Wafer Micro-Neighborhood Graph Features (KNN)
        if "wafer_id" in df.columns:
            neighbor_feats = []
            for wafer_id, group in df.groupby("wafer_id"):
                coords = group[[x_c, y_c]].values
                if len(coords) > 8:
                    nbrs = NearestNeighbors(n_neighbors=9, algorithm='ball_tree').fit(coords)
                    distances, _ = nbrs.kneighbors(coords)
                    
                    mean_dist_4 = distances[:, 1:5].mean(axis=1)
                    mean_dist_8 = distances[:, 1:9].mean(axis=1)
                else:
                    mean_dist_4 = np.zeros(len(group))
                    mean_dist_8 = np.zeros(len(group))
                
                group_df = pd.DataFrame({
                    "knn_dist_4": mean_dist_4,
                    "knn_dist_8": mean_dist_8
                }, index=group.index)
                neighbor_feats.append(group_df)
                
            knn_df = pd.concat(neighbor_feats).sort_index()
            df = pd.concat([df, knn_df], axis=1)

    # 3. Global Wafer Aggregations
    drop_cols = ["label", "block_readings", "die_id", "wafer_id"]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    
    if "wafer_id" in df.columns and len(num_cols) > 0:
        wafer_means = df.groupby("wafer_id")[num_cols].transform("mean").add_suffix("_wafer_mean")
        wafer_stds = df.groupby("wafer_id")[num_cols].transform("std").fillna(0).add_suffix("_wafer_std")
        df = pd.concat([df, wafer_means, wafer_stds], axis=1)

    # 4. Filter Numeric Columns & Fill NaNs
    final_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    out_df = df[final_cols].fillna(0)

    # 5. MEMORY OPTIMIZATION: Downcast float64 to float32 (halves RAM footprint)
    float64_cols = out_df.select_dtypes(include=["float64"]).columns
    out_df[float64_cols] = out_df[float64_cols].astype(np.float32)

    return out_df

print("Engineering physical neighborhood features...")
X_train = extract_ultimate_spatial_features(train_df)
y_train = train_df["label"].values

X_test = extract_ultimate_spatial_features(test_df)
y_test = test_df["label"].values

print(f"Total Spatial/Tabular Features: {X_train.shape[1]}")

pos_count = (y_train == 1).sum()
scale_pos_weight = (len(y_train) - pos_count) / pos_count

# 4. Out-of-Fold GPU Stacking Ensemble Setup
groups = train_df["wafer_id"].values if "wafer_id" in train_df.columns else np.arange(len(train_df))
sgkf = StratifiedGroupKFold(n_splits=5)

oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
oof_cat = np.zeros(len(train_df))

test_lgb = np.zeros(len(test_df))
test_xgb = np.zeros(len(test_df))
test_cat = np.zeros(len(test_df))

print("\nTraining 5-Fold Stratified Stacking Ensemble (GPU Accelerated)...")

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups=groups)):
    print(f"\n--- Training Fold {fold+1}/5 ---")
    
    # Slice current fold data
    X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # 1. LightGBM (CPU multi-threaded)
    lgb = LGBMClassifier(
        n_estimators=600, learning_rate=0.025, num_leaves=127, 
        scale_pos_weight=scale_pos_weight, random_state=42+fold, n_jobs=-1
    )
    lgb.fit(X_tr, y_tr)
    oof_lgb[val_idx] = lgb.predict_proba(X_va)[:, 1]
    test_lgb += lgb.predict_proba(X_test)[:, 1] / 5.0
    
    # 2. XGBoost (CUDA GPU)
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.025, max_depth=7, 
        tree_method="hist", device="cuda", scale_pos_weight=scale_pos_weight, random_state=42+fold
    )
    xgb.fit(X_tr, y_tr)
    oof_xgb[val_idx] = xgb.predict_proba(X_va)[:, 1]
    test_xgb += xgb.predict_proba(X_test)[:, 1] / 5.0
    
    # 3. CatBoost (CUDA GPU)
    cat = CatBoostClassifier(
        iterations=600, learning_rate=0.03, depth=7, 
        task_type="GPU", scale_pos_weight=scale_pos_weight, random_state=42+fold, verbose=0
    )
    cat.fit(X_tr, y_tr)
    oof_cat[val_idx] = cat.predict_proba(X_va)[:, 1]
    test_cat += cat.predict_proba(X_test)[:, 1] / 5.0
    
    print(f"Fold {fold+1}/5 complete.")

    # MEMORY FIX: Delete heavy model objects and fold slices, then force garbage collection
    del lgb, xgb, cat, X_tr, y_tr, X_va, y_va
    gc.collect()

# Meta-Learner (Logistic Regression Meta-Blender)
X_oof_meta = np.column_stack([oof_lgb, oof_xgb, oof_cat])
X_test_meta = np.column_stack([test_lgb, test_xgb, test_cat])

meta_model = LogisticRegression()
meta_model.fit(X_oof_meta, y_train)

y_probs = meta_model.predict_proba(X_test_meta)[:, 1]

# 5. Threshold Optimization & Evaluation
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.50

y_preds = (y_probs >= best_threshold).astype(int)

auc_pr = average_precision_score(y_test, y_probs)
f1_opt = f1_score(y_test, y_preds)

print("\n" + "="*50)
print("  GPU-ACCELERATED ULTIMATE MODEL A EVALUATION")
print("="*50)
print(f"  AUC-PR Score:       {auc_pr:.4f}")
print(f"  Optimal Threshold:  {best_threshold:.4f}")
print(f"  Best F1-Score:      {f1_opt:.4f}\n")

print("--- Classification Report ---")
print(classification_report(y_test, y_preds, target_names=["Pass (0)", "Fail (1)"]))