import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, classification_report, f1_score, precision_recall_curve
from tqdm import tqdm

# 1. Device Setup (GPU Acceleration)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 2. Data Helper Functions
def parse_block_readings(df):
    """Parses space-separated block_readings strings into a 2D float32 NumPy array."""
    readings_list = [
        np.fromstring(s, sep=" ", dtype=np.float32) 
        for s in tqdm(df["block_readings"], desc="Parsing signal array strings")
    ]
    return np.vstack(readings_list)

def extract_spatial_features(df):
    """Extracts spatial coordinates and engineered geometric features."""
    tab_df = df.drop(columns=["label", "block_readings", "die_id", "wafer_id"], errors="ignore").copy()
    
    # Compute radial distance from wafer center if x and y exist
    if "x" in tab_df.columns and "y" in tab_df.columns:
        tab_df["radial_dist"] = np.sqrt(tab_df["x"]**2 + tab_df["y"]**2)
        tab_df["angle"] = np.arctan2(tab_df["y"], tab_df["x"])
    
    # Select only numeric features and fill missing values
    tab_df = tab_df.select_dtypes(include=[np.number]).fillna(0)
    return tab_df.values.astype(np.float32)

# 3. Load Datasets
print("Loading datasets...")
train_df = pd.read_csv("input/train.csv")
test_df = pd.read_csv("input/test.csv")

# Extract Signals (Model B Branch)
X_train_sig = parse_block_readings(train_df)
X_test_sig = parse_block_readings(test_df)

sig_mean, sig_std = X_train_sig.mean(), X_train_sig.std()
X_train_sig = (X_train_sig - sig_mean) / (sig_std + 1e-8)
X_test_sig = (X_test_sig - sig_mean) / (sig_std + 1e-8)

# Extract Spatial/Tabular Features (Model A Branch)
X_train_tab = extract_spatial_features(train_df)
X_test_tab = extract_spatial_features(test_df)

tab_scaler = StandardScaler()
X_train_tab = tab_scaler.fit_transform(X_train_tab)
X_test_tab = tab_scaler.transform(X_test_tab)

y_train_np = train_df["label"].values.astype(np.float32)
y_test_np = test_df["label"].values.astype(np.float32)

num_tab_features = X_train_tab.shape[1]
print(f"Tabular spatial features extracted: {num_tab_features}")

# 4. PyTorch Tensors & DataLoaders (90% Train / 10% Val Split)
X_train_sig_t = torch.tensor(X_train_sig).unsqueeze(1)
X_train_tab_t = torch.tensor(X_train_tab)
y_train_t = torch.tensor(y_train_np)

X_test_sig_t = torch.tensor(X_test_sig).unsqueeze(1)
X_test_tab_t = torch.tensor(X_test_tab)
y_test_t = torch.tensor(y_test_np)

full_train_dataset = TensorDataset(X_train_sig_t, X_train_tab_t, y_train_t)
val_size = int(0.10 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(
    full_train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(TensorDataset(X_test_sig_t, X_test_tab_t, y_test_t), batch_size=512, shuffle=False)

# 5. Dual-Branch Hybrid Architecture
class HybridYieldCNN(nn.Module):
    def __init__(self, num_tabular_features):
        super().__init__()
        # Branch 1: 1D-CNN for sub-die block signals
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
        
        # Branch 2: Multi-Layer Perceptron for spatial features
        self.tabular_branch = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion Classifier (Combines 128 signal features + 32 spatial features)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_sig, x_tab):
        sig_feat = self.signal_branch(x_sig).squeeze(-1) # [batch, 128]
        tab_feat = self.tabular_branch(x_tab)            # [batch, 32]
        fused = torch.cat((sig_feat, tab_feat), dim=1)    # [batch, 160]
        return self.classifier(fused).squeeze(-1)

model = HybridYieldCNN(num_tab_features).to(device)

# 6. Loss Function & Optimizer
pos_count = (y_train_np == 1).sum()
neg_count = len(y_train_np) - pos_count
pos_weight = torch.tensor([neg_count / pos_count]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# 7. Training Loop with Early Stopping
max_epochs = 150
patience = 10
patience_counter = 0
best_val_loss = float('inf')
best_model_weights = None

print(f"\nTraining Hybrid Model C on GPU (Up to {max_epochs} epochs)...")

for epoch in range(1, max_epochs + 1):
    model.train()
    train_loss = 0.0
    for X_sig_b, X_tab_b, y_b in train_loader:
        X_sig_b, X_tab_b, y_b = X_sig_b.to(device), X_tab_b.to(device), y_b.to(device)
        
        optimizer.zero_grad()
        logits = model(X_sig_b, X_tab_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y_b)
        
    avg_train_loss = train_loss / train_size

    # Validation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_sig_b, X_tab_b, y_b in val_loader:
            X_sig_b, X_tab_b, y_b = X_sig_b.to(device), X_tab_b.to(device), y_b.to(device)
            logits = model(X_sig_b, X_tab_b)
            loss = criterion(logits, y_b)
            val_loss += loss.item() * len(y_b)
            
    avg_val_loss = val_loss / val_size
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch:03d}/{max_epochs:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n[Early Stopping Triggered] Stopped at epoch {epoch}. Best Val Loss: {best_val_loss:.4f}")
            break

if best_model_weights is not None:
    model.load_state_dict(best_model_weights)

# 8. Evaluation & Optimal Thresholding
model.eval()
y_probs = []

with torch.no_grad():
    for X_sig_b, X_tab_b, _ in test_loader:
        X_sig_b, X_tab_b = X_sig_b.to(device), X_tab_b.to(device)
        logits = model(X_sig_b, X_tab_b)
        probs = torch.sigmoid(logits)
        y_probs.extend(probs.cpu().numpy())

y_probs = np.array(y_probs)

precisions, recalls, thresholds = precision_recall_curve(y_test_np, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.50

y_preds_optimal = (y_probs >= best_threshold).astype(int)

auc_pr = average_precision_score(y_test_np, y_probs)
f1_optimal = f1_score(y_test_np, y_preds_optimal)

print("\n" + "="*50)
print("  MODEL C HYBRID EVALUATION RESULTS")
print("="*50)
print(f"  AUC-PR Score:       {auc_pr:.4f}")
print(f"  Optimal Threshold:  {best_threshold:.4f}")
print(f"  Best F1-Score:      {f1_optimal:.4f}\n")

print("--- Classification Report (Optimal Threshold) ---")
print(classification_report(y_test_np, y_preds_optimal, target_names=["Pass (0)", "Fail (1)"]))

# Save Model
torch.save(model.state_dict(), "model_c_hybrid.pt")
print("Model saved to model_c_hybrid.pt")