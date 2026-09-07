import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS for Vite React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dual-Branch Hybrid PyTorch Architecture ---
class HybridYieldCNN(nn.Module):
    def __init__(self, num_tabular_features):
        super().__init__()
        self.signal_branch = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_sig, x_tab):
        sig_feat = self.signal_branch(x_sig).squeeze(-1)
        tab_feat = self.tabular_branch(x_tab)
        fused = torch.cat((sig_feat, tab_feat), dim=1)
        return self.classifier(fused).squeeze(-1)

# --- Standalone Feature Extraction Helpers ---
def extract_ultimate_spatial_features(df):
    df = df.copy()
    xy_cols = [c for c in ["x", "y", "die_x", "die_y", "X", "Y"] if c in df.columns]
    
    if len(xy_cols) >= 2:
        x_c, y_c = xy_cols[0], xy_cols[1]
        df["radial_dist"] = np.sqrt(df[x_c]**2 + df[y_c]**2)
        df["angle"] = np.arctan2(df[y_c], df[x_c])
        df["manhattan_dist"] = np.abs(df[x_c]) + np.abs(df[y_c])
        df["wafer_ring_zone"] = pd.qcut(df["radial_dist"], q=10, labels=False, duplicates='drop')

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
                
                group_df = pd.DataFrame({"knn_dist_4": mean_dist_4, "knn_dist_8": mean_dist_8}, index=group.index)
                neighbor_feats.append(group_df)
            knn_df = pd.concat(neighbor_feats).sort_index()
            df = pd.concat([df, knn_df], axis=1)

    drop_cols = ["label", "block_readings", "die_id", "wafer_id"]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop_cols]
    
    if "wafer_id" in df.columns and len(num_cols) > 0:
        wafer_means = df.groupby("wafer_id")[num_cols].transform("mean").add_suffix("_wafer_mean")
        wafer_stds = df.groupby("wafer_id")[num_cols].transform("std").fillna(0).add_suffix("_wafer_std")
        df = pd.concat([df, wafer_means, wafer_stds], axis=1)

    final_cols = [c for c in df.columns if c not in drop_cols and np.issubdtype(df[c].dtype, np.number)]
    out_df = df[final_cols].fillna(0)
    float64_cols = out_df.select_dtypes(include=["float64"]).columns
    out_df[float64_cols] = out_df[float64_cols].astype(np.float32)
    return out_df

def extract_b_spatial(df):
    tab_df = df.drop(columns=["label", "block_readings", "die_id", "wafer_id"], errors="ignore").copy()
    if "x" in tab_df.columns and "y" in tab_df.columns:
        tab_df["radial_dist"] = np.sqrt(tab_df["x"]**2 + tab_df["y"]**2)
        tab_df["angle"] = np.arctan2(tab_df["y"], tab_df["x"])
    tab_df = tab_df.select_dtypes(include=[np.number]).fillna(0)
    return tab_df.values.astype(np.float32)

# Global Variables
model_a_bundle = None
model_b_net = None
model_b_scalers = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_models():
    global model_a_bundle, model_b_net, model_b_scalers
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # Load Model A
    model_a_bundle = joblib.load(os.path.join(models_dir, "model_a.pkl"))
    
    # Load Model B Scalers & Weights
    model_b_scalers = joblib.load(os.path.join(models_dir, "model_b_scalers.pkl"))
    num_tab_feats = model_b_scalers["num_tab_features"]
    
    model_b_net = HybridYieldCNN(num_tab_feats).to(device)
    model_b_net.load_state_dict(torch.load(os.path.join(models_dir, "model_b.pt"), map_location=device))
    model_b_net.eval()
    print("All ML models loaded successfully into FastAPI!")

@app.post("/predict")
async def predict_wafer(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
        
        # --- Model A Inference ---
        X_a = extract_ultimate_spatial_features(df)
        feature_names_a = model_a_bundle["feature_names"]
        for col in feature_names_a:
            if col not in X_a.columns:
                X_a[col] = 0
        X_a = X_a[feature_names_a]
        
        p_lgb = np.mean([m.predict_proba(X_a)[:, 1] for m in model_a_bundle["lgb_models"]], axis=0)
        p_xgb = np.mean([m.predict_proba(X_a)[:, 1] for m in model_a_bundle["xgb_models"]], axis=0)
        p_cat = np.mean([m.predict_proba(X_a)[:, 1] for m in model_a_bundle["cat_models"]], axis=0)
        
        X_meta = np.column_stack([p_lgb, p_xgb, p_cat])
        fail_probs_a = model_a_bundle["meta_model"].predict_proba(X_meta)[:, 1]
        
        # --- Model B Inference ---
        readings_list = [np.fromstring(s, sep=" ", dtype=np.float32) for s in df["block_readings"]]
        X_sig = np.vstack(readings_list)
        
        sig_mean = model_b_scalers["sig_mean"]
        sig_std = model_b_scalers["sig_std"]
        X_sig_norm = (X_sig - sig_mean) / (sig_std + 1e-8)
        
        X_tab_b = extract_b_spatial(df)
        X_tab_b_scaled = model_b_scalers["tab_scaler"].transform(X_tab_b)
        
        t_sig = torch.tensor(X_sig_norm, dtype=torch.float32).unsqueeze(1).to(device)
        t_tab = torch.tensor(X_tab_b_scaled, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            logits_b = model_b_net(t_sig, t_tab)
            fail_probs_b = torch.sigmoid(logits_b).cpu().numpy()
            
        results = []
        for i, row in df.iterrows():
            results.append({
                "die_id": int(row.get("die_id", i + 1)),
                "x": int(row.get("x", 0)),
                "y": int(row.get("y", 0)),
                "fail_prob_A": float(fail_probs_a[i]),
                "fail_prob_B": float(fail_probs_b[i]),
                "label": int(row.get("label", 0))
            })
            
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))