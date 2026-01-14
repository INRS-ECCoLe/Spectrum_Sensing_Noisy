import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import os
import warnings
import collections
import csv
import gc
import pandas as pd

# Filter UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'sample_len': 1024,      # HisarMod2019 length
    'n_channels': 2,        
    # Dataset Files
    'train_data_path': os.environ.get('HISAR_TRAIN_DATA', 'HisarMod2019/train_data.csv'),
    'train_snr_path': os.environ.get('HISAR_TRAIN_SNR', 'HisarMod2019/train_snr.csv'),
    'test_data_path': os.environ.get('HISAR_TEST_DATA', 'HisarMod2019/test_data.csv'),
    'test_snr_path': os.environ.get('HISAR_TEST_SNR', 'HisarMod2019/test_snr.csv'),
    
    'batch_size': 256,
    'epochs': 150,
    'learning_rate': 0.001,
    'results_dir': 'results_hisarmod_sensing',
    'model_base_path': 'hisarmod_model_',
    # Callback Settings
    'early_stopping_patience': 15,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-6,
    # Resource Settings
    'num_workers': 8, 
    'pin_memory': True,
    # Data Limiter (HisarMod is huge ~780k. 50k-100k recommended for testing on limited RAM)
    'subsample_limit': 100000 
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {DEVICE}")

# ==========================================
# DATA LOADING (CSV PAIRS)
# ==========================================
def load_hisarmod_subset(data_path, snr_path, limit=None, is_test=False):
    """
    Loads a subset (Train or Test) from CSV files.
    Returns X, y, snr.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    if not os.path.exists(snr_path):
        raise FileNotFoundError(f"SNR file not found at {snr_path}")

    subset_name = "TEST" if is_test else "TRAIN"
    print(f"\n[Data] Loading {subset_name} subset...")

    # --- Load SNR Labels ---
    print(f"[Data] Reading SNR labels from {snr_path}...")
    try:
        # Load SNR. Assuming simple CSV with one column or 'snr' header
        df_snr = pd.read_csv(snr_path, nrows=limit)
        # Check if it has a header or just values
        if 'snr' in [c.lower() for c in df_snr.columns]:
            # Find the column that looks like 'snr'
            col = next(c for c in df_snr.columns if c.lower().strip() == 'snr')
            snr_full = df_snr[col].values
        else:
            # Assume first column is SNR if no header match
            snr_full = df_snr.iloc[:, 0].values
            
        print(f"[Data] Found {len(snr_full)} SNR labels.")
    except Exception as e:
        print(f"[Error] Failed to load SNR CSV: {e}")
        raise e

    # --- Load IQ Data ---
    # Load matching number of data rows
    n_to_load = len(snr_full)
    print(f"[Data] Reading IQ Data from {data_path} (Limit: {n_to_load})...")
    
    try:
        # Load IQ data
        df_data = pd.read_csv(data_path, nrows=n_to_load)
        
        # --- Robust Type Conversion (Fix for 'str' error) ---
        # Check if pandas loaded data as objects (strings)
        if df_data.select_dtypes(include=['object']).shape[1] > 0:
            print("[Data] Detected non-numeric data (strings). Attempting conversion...")
            # Convert values to numpy array of strings first
            vals = df_data.values.astype(str)
            
            # Check for MATLAB-style complex format (e.g., '1.0000+2.0000i') by checking first element
            sample_val = vals[0, 0] if vals.size > 0 else ""
            if 'i' in sample_val or 'j' in sample_val:
                print("[Data] Converting complex strings (MATLAB format 'i' -> 'j')...")
                vals = np.char.replace(vals, 'i', 'j') # Python requires 'j' for complex
                vals = np.char.replace(vals, ' ', '')  # Remove any accidental spaces
                X_raw = vals.astype(np.complex128)
            else:
                print("[Data] Converting string floats...")
                X_raw = vals.astype(np.float32)
        else:
            # Already numeric
            X_raw = df_data.values
        
        # Cleanup DataFrame to free memory
        del df_data
        gc.collect()
        
    except Exception as e:
        print(f"[Error] Failed to load Data CSV: {e}")
        raise e

    # --- Alignment Check ---
    if len(X_raw) != len(snr_full):
        min_len = min(len(X_raw), len(snr_full))
        print(f"[Warning] Mismatch in {subset_name}: Data {len(X_raw)} vs SNR {len(snr_full)}. Trimming to {min_len}.")
        X_raw = X_raw[:min_len]
        snr_full = snr_full[:min_len]

    # --- Reshaping ---
    # Expected: (N, 1024, 2)
    # Incoming likely: (N, 2048) or (N, 2, 1024)
    # HisarMod: 1024 samples. 1024 * 2 = 2048 features.
    
    n_samples = X_raw.shape[0]
    n_features = X_raw.shape[1]
    
    target_len = CONFIG['sample_len'] # 1024
    
    if n_features == target_len * 2:
        # Likely flattened I/Q. Reshape to (N, 1024, 2)
        print(f"[Data] Reshaping flattened data {X_raw.shape} to (N, {target_len}, 2)...")
        # Assume first half is I, second half is Q (common export format)
        I = X_raw[:, :target_len]
        Q = X_raw[:, target_len:]
        
        # Ensure float32 if we have separated real components
        if np.iscomplexobj(I): I = I.real
        if np.iscomplexobj(Q): Q = Q.real
        
        X_h1 = np.stack([I, Q], axis=2).astype(np.float32) # (N, 1024, 2)
        
    elif n_features == target_len and np.iscomplexobj(X_raw):
        # Complex support
        print(f"[Data] Splitting complex data {X_raw.shape} into Real/Imag channels...")
        X_h1 = np.stack((X_raw.real, X_raw.imag), axis=2).astype(np.float32)
    else:
        # Fallback reshape
        print(f"[Warning] Data shape {X_raw.shape} doesn't match expected standard. Attempting best-effort reshape.")
        if n_features % 2 == 0:
             new_len = n_features // 2
             CONFIG['sample_len'] = new_len
             X_h1 = X_raw.reshape(n_samples, new_len, 2).astype(np.float32)
        else:
             raise ValueError(f"Cannot reshape {n_features} features into I/Q (must be even).")

    print(f"[Data] Loaded {X_h1.shape[0]} H1 (Signal) samples.")

    # --- Generate H0 (Noise) ---
    print(f"[Data] Generating {n_samples} synthetic H0 (Noise) samples...")
    X_h0 = np.random.normal(0, 1.0, X_h1.shape).astype(np.float32)
    
    y_h1 = np.ones(n_samples)
    y_h0 = np.zeros(n_samples)
    
    snr_h1 = snr_full
    snr_h0 = snr_full.copy()
    
    # Combine
    X = np.concatenate([X_h0, X_h1], axis=0)
    y = np.concatenate([y_h0, y_h1], axis=0)
    snr = np.concatenate([snr_h0, snr_h1], axis=0)
    
    return X, y, snr

# ==========================================
# PYTORCH MODEL DEFINITIONS
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_classes=2):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :] 
        x = self.fc(x)
        return x 

class GRUModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_classes=2):
        super(GRUModel, self).__init__()
        self.gru1 = nn.GRU(input_size, hidden_size, batch_first=True)
        self.gru2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

class CNN2Model(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN2Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(2, 8), padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(1, 2)) 
        self.drop = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(2, 8), padding='same')
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(2, 8), padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(2, 8), padding='same')
        
        # Dynamic Flatten Calculation
        # 4 pooling layers of kernel (1,2) -> Length reduction factor 16
        final_len = CONFIG['sample_len'] // 16
        if final_len < 1: final_len = 1
        
        self.fc1 = nn.Linear(64 * 2 * final_len, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1) 
        x = self.drop(self.pool(self.relu(self.conv1(x))))
        x = self.drop(self.pool(self.relu(self.conv2(x))))
        x = self.drop(self.pool(self.relu(self.conv3(x))))
        x = self.drop(self.pool(self.relu(self.conv4(x))))
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PETCGDNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PETCGDNNModel, self).__init__()
        self.fc_geo = nn.Linear(CONFIG['sample_len'] * 2, 1) 
        self.conv1 = nn.Conv2d(1, 75, kernel_size=(8, 2)) 
        self.conv2 = nn.Conv2d(75, 25, kernel_size=(5, 1))
        self.gru = nn.GRU(25, 128, batch_first=True)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        flat = x.flatten(start_dim=1)
        linear_x = self.fc_geo(flat)
        cos_x = torch.cos(linear_x).unsqueeze(-1)
        sin_x = torch.sin(linear_x).unsqueeze(-1)
        
        inp_i = x[:, :, 0].unsqueeze(-1)
        inp_q = x[:, :, 1].unsqueeze(-1)
        
        x11 = inp_i * cos_x
        x12 = inp_q * sin_x
        x21 = inp_q * cos_x
        x22 = inp_i * sin_x
        
        y1 = x11 + x12
        y2 = x21 - x22
        
        x_geo = torch.cat([y1, y2], dim=2).unsqueeze(1) 
        
        c1 = torch.relu(self.conv1(x_geo)) 
        c2 = torch.relu(self.conv2(c1))
        
        gru_in = c2.squeeze(-1).permute(0, 2, 1)
        gru_out, _ = self.gru(gru_in)
        last_step = gru_out[:, -1, :]
        return self.out(last_step)

# ==========================================
# TRAINING & EVALUATION HELPERS
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start = time.time()
    
    for inputs, labels, _ in loader: 
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    duration = time.time() - start
    return running_loss / total, correct / total, duration

def evaluate_by_snr(model, loader):
    model.eval()
    all_preds, all_labels, all_probs, all_snrs = [], [], [], []
    
    start_time = time.time()
    with torch.no_grad():
        for inputs, labels, snrs in loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_snrs.extend(snrs.numpy())
            
    total_time = time.time() - start_time
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_snrs = np.array(all_snrs)
    
    unique_snrs = sorted(np.unique(all_snrs))
    results_by_snr = collections.defaultdict(dict)
    
    from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

    for snr in unique_snrs:
        mask = (all_snrs == snr)
        y_true_s = all_labels[mask]
        y_pred_s = all_preds[mask]
        y_prob_s = all_probs[mask]
        
        if len(y_true_s) == 0: continue

        tp = np.sum((y_true_s == 1) & (y_pred_s == 1))
        tn = np.sum((y_true_s == 0) & (y_pred_s == 0))
        fp = np.sum((y_true_s == 0) & (y_pred_s == 1))
        fn = np.sum((y_true_s == 1) & (y_pred_s == 0))

        precision = precision_score(y_true_s, y_pred_s, zero_division=0)
        recall = recall_score(y_true_s, y_pred_s, zero_division=0)
        f1 = f1_score(y_true_s, y_pred_s, zero_division=0)
        accuracy = (tp + tn) / len(y_true_s)
        pfa = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        
        # NMSE
        y_prob_class1 = y_prob_s[:, 1]
        numerator = np.sum((y_true_s - y_prob_class1)**2)
        denominator = np.sum(y_true_s**2)
        nmse = numerator / (denominator + 1e-10)

        # AUC
        try:
            if len(np.unique(y_true_s)) > 1:
                fpr, tpr, _ = roc_curve(y_true_s, y_prob_s[:, 1])
                roc_auc = auc(fpr, tpr)
            else:
                roc_auc = 0.5
        except ValueError:
            roc_auc = 0.5
            
        results_by_snr[snr] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pd': recall,
            'pfa': pfa,
            'nmse': nmse,
            'auc': roc_auc
        }

    return results_by_snr, (all_labels, all_probs)

def run_experiment(model_name, train_loader, test_loader):
    print(f"\n[Model] Starting {model_name}...")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if model_name == 'LSTM': model = LSTMModel().to(DEVICE)
    elif model_name == 'GRU': model = GRUModel().to(DEVICE)
    elif model_name == 'CNN': model = CNN2Model().to(DEVICE)
    elif model_name == 'PETCGDNN': model = PETCGDNNModel().to(DEVICE)
    else: raise ValueError("Unknown model")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['lr_scheduler_factor'], 
        patience=CONFIG['lr_scheduler_patience'], min_lr=CONFIG['min_lr']
    )
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc, duration = train_one_epoch(model, train_loader, criterion, optimizer)
        
        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i, l, _ in test_loader:
                i, l = i.to(DEVICE), l.to(DEVICE)
                out = model(i)
                loss = criterion(out, l)
                running_val_loss += loss.item() * i.size(0)
                _, pred = torch.max(out, 1)
                total += l.size(0)
                correct += (pred == l).sum().item()
        
        val_loss = running_val_loss / total
        val_acc = correct / total
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['results_dir'], f"{CONFIG['model_base_path']}{model_name}_best.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping.")
                break
    
    model.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], f"{CONFIG['model_base_path']}{model_name}_best.pth")))
    snr_results, (y_all, probs_all) = evaluate_by_snr(model, test_loader)
    
    return snr_results

def save_consolidated_csv(all_results):
    filepath = os.path.join(CONFIG['results_dir'], 'all_models_snr_metrics_hisarmod.csv')
    
    if not all_results:
        print("No results to save.")
        return

    first_model = list(all_results.keys())[0]
    first_snr = list(all_results[first_model].keys())[0]
    metric_keys = list(all_results[first_model][first_snr].keys())
    
    header = ['Model', 'SNR'] + metric_keys
    
    print(f"[Data] Saving consolidated metrics to {filepath}...")
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model_name, snr_dict in all_results.items():
            sorted_snrs = sorted(snr_dict.keys(), key=lambda x: float(x))
            for snr in sorted_snrs:
                metrics = snr_dict[snr]
                row = [model_name, snr] + [metrics[k] for k in metric_keys]
                writer.writerow(row)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    try:
        # Load Train Data
        X_train, y_train, snr_train = load_hisarmod_subset(
            CONFIG['train_data_path'], 
            CONFIG['train_snr_path'], 
            limit=CONFIG['subsample_limit'], 
            is_test=False
        )
        
        # Load Test Data
        X_test, y_test, snr_test = load_hisarmod_subset(
            CONFIG['test_data_path'], 
            CONFIG['test_snr_path'], 
            limit=CONFIG['subsample_limit'], 
            is_test=True
        )
        
        # Normalize (Using Train stats usually preferred, here per-subset for simplicity)
        print("[Data] Normalizing...")
        energy_train = np.sum(np.abs(X_train)**2, axis=(1, 2), keepdims=True)
        X_train = X_train / (np.sqrt(energy_train / (CONFIG['sample_len']*2)) + 1e-10)
        
        energy_test = np.sum(np.abs(X_test)**2, axis=(1, 2), keepdims=True)
        X_test = X_test / (np.sqrt(energy_test / (CONFIG['sample_len']*2)) + 1e-10)
        
        # To Tensor
        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32), 
            torch.tensor(y_train, dtype=torch.long), 
            torch.tensor(snr_train, dtype=torch.float32)
        )
        test_ds = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32), 
            torch.tensor(y_test, dtype=torch.long), 
            torch.tensor(snr_test, dtype=torch.float32)
        )
        
        # DataLoaders
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
        
        # Order: LSTM, GRU, CNN, PETCGDNN
        models_list = ['LSTM', 'GRU', 'CNN', 'PETCGDNN']
        all_model_results = {}
        
        for name in models_list:
            res = run_experiment(name, train_loader, test_loader)
            all_model_results[name] = res
            
            # Cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        save_consolidated_csv(all_model_results)
        print("[Success] All HisarMod2019 results exported.")
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()