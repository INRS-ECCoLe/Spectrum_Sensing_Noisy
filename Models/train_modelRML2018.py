import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import time
import os
import json
import warnings
import collections
import csv
import h5py  # Required for RML2018

# Filter UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'sample_len': 1024,      # RML2018 length
    'n_channels': 2,        
    'dataset_path': os.environ.get('RML2018_PATH', 'RML2018.hdf5'),
    'batch_size': 256,
    'epochs': 150,           # Restored to 150
    'learning_rate': 0.001,
    'test_size': 0.2,        # Restored explicit test size
    'results_dir': 'results_narval_snr_analysis_RML2018', # Restored specific dir name
    'model_base_path': 'rml2018_model_',
    # Callback Settings (Restored)
    'early_stopping_patience': 15,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-6,
    # Resource Settings
    'num_workers': 8, 
    'pin_memory': True,
    # Data Limiter (CRITICAL for RML2018 to avoid OOM on standard RAM)
    'subsample_limit': 100000 
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {DEVICE}")

# ==========================================
# DATA LOADING (RML2018 SPECIFIC)
# ==========================================
def load_rml2018_dataset(path, limit=None):
    """
    Loads RML2018 dataset (HDF5 format).
    RML2018 contains only H1 (Signals). We generate H0 (Noise).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please download GOLD_XYZ_OSC.0001_1024.hdf5")

    print(f"[Data] Opening RML2018 HDF5 from {path}...")
    
    with h5py.File(path, 'r') as f:
        # RML2018 Structure: X (Data), Y (Labels), Z (SNR)
        # Full shape is approx (2.5M, 1024, 2)
        
        total_samples = f['X'].shape[0]
        n_load = total_samples if limit is None else min(limit, total_samples)
        
        print(f"[Data] Loading {n_load} samples (H1)...")
        
        # Load H1 (Signal)
        # X shape in file is (N, 1024, 2)
        X_h1 = f['X'][:n_load] 
        snr_h1 = f['Z'][:n_load]
        
        # Squeeze SNR if it's shape (N, 1)
        if len(snr_h1.shape) > 1:
            snr_h1 = snr_h1.flatten()
            
        y_h1 = np.ones(n_load) # Label 1 for Signal Present
        
    print(f"[Data] Loaded {X_h1.shape[0]} H1 samples.")
    
    # Generate H0 (Noise) to balance the dataset
    # We create a noise-only dataset matching dimensions and SNR labels
    # (SNR label on noise implies "Noise power equivalent to Signal at that SNR")
    print(f"[Data] Generating {n_load} synthetic H0 samples...")
    
    # RML2018 data is roughly unit energy, noise power varies.
    # However, for pure noise generation (H0), we simulate AWGN.
    # We approximate noise floor based on dataset characteristics or set a fixed low power.
    # Here we generate standard normal noise.
    
    X_h0 = np.random.normal(0, 1.0, X_h1.shape) 
    
    # Scale H0 to match the noise floor found in low-SNR RML samples (approx)
    # Or simply normalize both later.
    
    y_h0 = np.zeros(n_load) # Label 0 for Noise Only
    snr_h0 = snr_h1.copy()  # Copy SNR labels to allow per-SNR evaluation
    
    # Combine
    X = np.concatenate([X_h0, X_h1], axis=0)
    y = np.concatenate([y_h0, y_h1], axis=0)
    snr = np.concatenate([snr_h0, snr_h1], axis=0)
    
    return X, y, snr

# ==========================================
# PYTORCH MODEL DEFINITIONS (SCALED FOR 1024)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_classes=2):
        super(LSTMModel, self).__init__()
        # Input: (Batch, 1024, 2)
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
        # Input assumes (Batch, 1, 2, 1024) after permute
        # Kernel (2, 8) convolves over (Channel, Time)
        
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(2, 8), padding='same')
        self.pool = nn.MaxPool2d(kernel_size=(1, 2)) # Pools time dim by 2
        self.drop = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(256, 128, kernel_size=(2, 8), padding='same')
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(2, 8), padding='same')
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(2, 8), padding='same')
        
        # Flatten sizing calculation for length 1024:
        # 1024 -> pool -> 512 -> pool -> 256 -> pool -> 128 -> pool -> 64
        # Final shape: (Batch, 64_channels, 2_height, 64_width)
        self.fc1 = nn.Linear(64 * 2 * 64, 128) 
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x input: (Batch, 1024, 2)
        # Permute to (Batch, 1, 2, 1024) for Conv2d treating 2 as Height
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
        # Scaled from 128 to 1024
        self.fc_geo = nn.Linear(1024 * 2, 1) 
        self.conv1 = nn.Conv2d(1, 75, kernel_size=(8, 2)) 
        self.conv2 = nn.Conv2d(75, 25, kernel_size=(5, 1))
        self.gru = nn.GRU(25, 128, batch_first=True)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (Batch, 1024, 2)
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
        
        x_geo = torch.cat([y1, y2], dim=2).unsqueeze(1) # (B, 1, 1024, 2)
        
        c1 = torch.relu(self.conv1(x_geo)) # Conv over (8,2)
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
    
    for snr in unique_snrs:
        mask = (all_snrs == snr)
        y_true_s = all_labels[mask]
        y_pred_s = all_preds[mask]
        y_prob_s = all_probs[mask]
        
        if len(y_true_s) == 0: continue

        # Handle confusion matrix shape issues
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(np.unique(y_true_s)) > 1 or len(np.unique(y_pred_s)) > 1:
             cm = confusion_matrix(y_true_s, y_pred_s, labels=[0, 1])
             tn, fp, fn, tp = cm.ravel()
        else:
             # Manual calculation for edge cases
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
            'recall': recall, # Pd
            'f1': f1,
            'pd': recall,
            'pfa': pfa,
            'nmse': nmse,
            'auc': roc_auc
        }

    return results_by_snr, (all_labels, all_probs)

def run_experiment(model_name, train_loader, test_loader):
    print(f"\n[Model] Starting {model_name}...")
    
    # Force GPU cleanup before allocating new model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if model_name == 'LSTM': model = LSTMModel().to(DEVICE)
    elif model_name == 'GRU': model = GRUModel().to(DEVICE)
    elif model_name == 'CNN': model = CNN2Model().to(DEVICE)
    elif model_name == 'PETCGDNN': model = PETCGDNNModel().to(DEVICE)
    else: raise ValueError("Unknown model")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Restored LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=CONFIG['lr_scheduler_factor'], 
        patience=CONFIG['lr_scheduler_patience'], min_lr=CONFIG['min_lr']
    )
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc, duration = train_one_epoch(model, train_loader, criterion, optimizer)
        
        # Validation for Callbacks
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
        
        # Scheduler Step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.2e}")
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['results_dir'], f"{CONFIG['model_base_path']}{model_name}_best.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping.")
                break
    
    # Load Best
    model.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], f"{CONFIG['model_base_path']}{model_name}_best.pth")))
    snr_results, (y_all, probs_all) = evaluate_by_snr(model, test_loader)
    
    return snr_results

def save_consolidated_csv(all_results):
    filepath = os.path.join(CONFIG['results_dir'], 'all_models_snr_metrics_rml2018.csv')
    
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
        # Load Data
        X, y, snr = load_rml2018_dataset(CONFIG['dataset_path'], limit=CONFIG['subsample_limit'])
        
        # Normalize
        print("[Data] Normalizing...")
        energy = np.sum(np.abs(X)**2, axis=(1, 2), keepdims=True)
        X = X / (np.sqrt(energy / (CONFIG['sample_len']*2)) + 1e-10)
        
        # To Tensor
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        snr_t = torch.tensor(snr, dtype=torch.float32)
        
        dataset = TensorDataset(X_t, y_t, snr_t)
        
        # Updated Split using CONFIG['test_size']
        train_size_len = int((1 - CONFIG['test_size']) * len(dataset))
        test_size_len = len(dataset) - train_size_len
        train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size_len, test_size_len])
        
        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
        test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory'])
        
        models_list = ['LSTM', 'GRU', 'CNN', 'PETCGDNN']
        all_model_results = {}
        
        for name in models_list:
            res = run_experiment(name, train_loader, test_loader)
            all_model_results[name] = res
            
            # Explicit cleanup after each experiment
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        save_consolidated_csv(all_model_results)
        print("[Success] All RML2018 results exported.")
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")
    except Exception as e:
        print(f"[Error] An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()