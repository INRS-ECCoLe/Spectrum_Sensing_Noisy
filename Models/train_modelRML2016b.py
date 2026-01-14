import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
import scipy.signal
import time
import os
import json
import pickle
import warnings
import collections
import csv

# Filter UserWarnings as requested
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIGURATION
# ==========================================
CONFIG = {
    'sample_len': 128,      
    'n_channels': 2,        
    'dataset_path': os.environ.get('HDF5_DATA_PATH', 'RML2016b.dat'),
    'batch_size': 256,
    'epochs': 150,
    'learning_rate': 0.001,
    'test_size': 0.2,
    'results_dir': 'results_narval_snr_analysis_RML2016b',
    'model_base_path': 'pytorch_model_',
    # Callback Settings
    'early_stopping_patience': 15,
    'lr_scheduler_patience': 5,
    'lr_scheduler_factor': 0.5,
    'min_lr': 1e-6,
    # Resource Settings (Matches SBATCH --cpus-per-task)
    'num_workers': 8, # Uses multiple CPUs for data loading
    'pin_memory': True # Speeds up host-to-device transfer
}

os.makedirs(CONFIG['results_dir'], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Info] Using device: {DEVICE}")

# ==========================================
# DATA LOADING (SNR AWARE)
# ==========================================
def load_rml2016_dataset(path):
    """
    Loads RML2016 dataset and preserves SNR labels.
    Returns: X, y, snr_labels
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please upload 'RML2016a.dat'.")

    print(f"[Data] Loading RML2016 dataset from {path}...")
    
    with open(path, 'rb') as f:
        Xd = pickle.load(f, encoding='latin1')

    X_list = []
    y_list = []
    snr_list = []
    
    # RML structure is Dict{(Modulation, SNR): DataArray}
    # We aggregate all modulations, but keep track of SNRs
    for (mod, snr), data in Xd.items():
        # data shape: (N, 2, 128) -> Transpose to (N, 128, 2)
        data = np.transpose(data, (0, 2, 1))
        
        n_samples = data.shape[0]
        
        X_list.append(data)
        y_list.append(np.ones(n_samples)) # Label 1 (H1)
        snr_list.append(np.full(n_samples, snr))
        
    X_h1 = np.vstack(X_list)
    y_h1 = np.concatenate(y_list)
    snr_h1 = np.concatenate(snr_list)
    
    n_samples_h1 = X_h1.shape[0]
    print(f"[Data] Loaded {n_samples_h1} H1 samples.")
    
    # Generate H0 (Noise) to balance the dataset
    # We assign H0 samples the same SNR labels as the H1 samples 
    # to allow "Per SNR" evaluation (detecting signal vs noise at that specific SNR level)
    print(f"[Data] Generating {n_samples_h1} synthetic H0 samples...")
    noise_power = 0.001
    X_h0 = np.random.normal(0, np.sqrt(noise_power), (n_samples_h1, 128, 2))
    y_h0 = np.zeros(n_samples_h1)
    snr_h0 = snr_h1.copy() # Match SNRs for balanced evaluation buckets
    
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
        self.fc1 = nn.Linear(64 * 2 * 8, 128) 
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

class MCLDNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MCLDNNModel, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 50, kernel_size=(2, 8), padding='same')
        self.conv1_2 = nn.Conv1d(1, 50, kernel_size=8, padding='same')
        self.conv1_3 = nn.Conv1d(1, 50, kernel_size=8, padding='same')
        self.fusion_fc = nn.Linear(50 * 2 * 128 + 50 * 128 + 50 * 128, 124 * 100)
        self.lstm1 = nn.LSTM(100, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_classes)
        self.selu = nn.SELU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        i1 = x.permute(0, 2, 1).unsqueeze(1)
        i2 = x[:, :, 0].unsqueeze(1)
        i3 = x[:, :, 1].unsqueeze(1)
        x1 = torch.relu(self.conv1_1(i1))
        x2 = torch.relu(self.conv1_2(i2))
        x3 = torch.relu(self.conv1_3(i3))
        f1 = x1.flatten(start_dim=1)
        f2 = x2.flatten(start_dim=1)
        f3 = x3.flatten(start_dim=1)
        merged = torch.cat([f1, f2, f3], dim=1)
        merged = self.fusion_fc(merged)
        merged = merged.view(-1, 124, 100)
        lstm_out, _ = self.lstm1(merged)
        lstm_out, _ = self.lstm2(lstm_out)
        lstm_last = lstm_out[:, -1, :]
        d = self.drop(self.selu(self.fc1(lstm_last)))
        d = self.drop(self.selu(self.fc2(d)))
        return self.out(d)

class PETCGDNNModel(nn.Module):
    def __init__(self, num_classes=2):
        super(PETCGDNNModel, self).__init__()
        self.fc_geo = nn.Linear(128 * 2, 1) 
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
    
    for inputs, labels, _ in loader: # Ignore SNR during training
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

def evaluate_loss_acc(model, loader, criterion):
    # Quick eval for callbacks (ignoring SNR breakdown)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _ in loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, correct / total

def evaluate_by_snr(model, loader):
    """
    Evaluates model and returns metrics broken down by SNR.
    """
    model.eval()
    
    # Store all results
    all_preds = []
    all_labels = []
    all_probs = []
    all_snrs = []
    
    # Inference
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
    
    # Convert to numpy
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_snrs = np.array(all_snrs)
    
    unique_snrs = sorted(np.unique(all_snrs))
    results_by_snr = collections.defaultdict(dict)
    
    # Calculate metrics per SNR
    for snr in unique_snrs:
        mask = (all_snrs == snr)
        y_true_s = all_labels[mask]
        y_pred_s = all_preds[mask]
        y_prob_s = all_probs[mask]
        
        # Avoid calculation errors if batch is empty (rare)
        if len(y_true_s) == 0: continue

        cm = confusion_matrix(y_true_s, y_pred_s)
        # Handle shape if only 1 class present
        if cm.shape == (1, 1):
            # Try to infer based on labels
            tn, fp, fn, tp = 0, 0, 0, 0
            if y_true_s[0] == 0: 
                tn = cm[0,0] if y_pred_s[0] == 0 else 0
                fp = cm[0,0] if y_pred_s[0] == 1 else 0
            else:
                tp = cm[0,0] if y_pred_s[0] == 1 else 0
                fn = cm[0,0] if y_pred_s[0] == 0 else 0
        else:
            tn, fp, fn, tp = cm.ravel()

        # Metrics
        precision = precision_score(y_true_s, y_pred_s, zero_division=0)
        recall = recall_score(y_true_s, y_pred_s, zero_division=0) # Pd
        f1 = f1_score(y_true_s, y_pred_s, zero_division=0)
        accuracy = (tp + tn) / len(y_true_s)
        
        pd = recall # Probability of Detection
        pfa = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        
        # NMSE
        y_prob_class1 = y_prob_s[:, 1]
        numerator = np.sum((y_true_s - y_prob_class1)**2)
        denominator = np.sum(y_true_s**2)
        nmse = numerator / (denominator + 1e-10)

        # AUC
        try:
            fpr, tpr, _ = roc_curve(y_true_s, y_prob_s[:, 1])
            roc_auc = auc(fpr, tpr)
        except ValueError:
            roc_auc = 0.5 # Default if only one class exists
            
        results_by_snr[snr] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'pd': pd,
            'pfa': pfa,
            'nmse': nmse,
            'auc': roc_auc
        }

    return results_by_snr, total_time, (all_labels, all_probs)

# ==========================================
# MAIN EXECUTION
# ==========================================
def run_experiment(model_name, train_loader, test_loader):
    print(f"\n[Model] Starting {model_name}...")
    
    if model_name == 'LSTM': model = LSTMModel().to(DEVICE)
    elif model_name == 'GRU': model = GRUModel().to(DEVICE)
    elif model_name == 'CNN': model = CNN2Model().to(DEVICE)
    elif model_name == 'MCLDNN': model = MCLDNNModel().to(DEVICE)
    elif model_name == 'PETCGDNN': model = PETCGDNNModel().to(DEVICE)
    else: raise ValueError("Unknown model")
        
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    # Removed verbose=True due to TypeError in newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=CONFIG['lr_scheduler_factor'], patience=CONFIG['lr_scheduler_patience'], min_lr=CONFIG['min_lr'])
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    history_data = []
    
    # Train
    for epoch in range(CONFIG['epochs']):
        train_loss, train_acc, duration = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate_loss_acc(model, test_loader, criterion)
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} - Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f} - LR: {current_lr:.2e}")
        
        history_data.append({'epoch': epoch+1, 'loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(CONFIG['results_dir'], f"{CONFIG['model_base_path']}{model_name}_best.pth"))
        else:
            early_stop_counter += 1
            if early_stop_counter >= CONFIG['early_stopping_patience']:
                print("Early stopping.")
                break
    
    # Load Best and Detailed Eval
    model.load_state_dict(torch.load(os.path.join(CONFIG['results_dir'], f"{CONFIG['model_base_path']}{model_name}_best.pth")))
    
    snr_results, inf_time, (y_all, probs_all) = evaluate_by_snr(model, test_loader)
    
    # Save Results
    with open(os.path.join(CONFIG['results_dir'], f'results_snr_{model_name}.json'), 'w') as f:
        # Convert keys to str for JSON
        serializable_res = {str(k): v for k, v in snr_results.items()}
        json.dump(serializable_res, f, indent=4)

    # Save raw ROC data for later plotting
    fpr, tpr, _ = roc_curve(y_all, probs_all[:, 1])
    roc_stack = np.column_stack((fpr, tpr))
    np.savetxt(os.path.join(CONFIG['results_dir'], f'roc_data_{model_name}.csv'), roc_stack, delimiter=",", header="FPR,TPR", comments="")
        
    return snr_results, (y_all, probs_all)

def save_consolidated_csv(all_results):
    """
    Saves a single CSV with all metrics for all models at all SNRs.
    Columns: Model, SNR, Accuracy, Precision, Recall, F1, Pd, Pfa, NMSE, AUC
    """
    filepath = os.path.join(CONFIG['results_dir'], 'all_models_snr_metrics.csv')
    
    # Get all metric keys from the first entry to define columns
    first_model = list(all_results.keys())[0]
    first_snr = list(all_results[first_model].keys())[0]
    metric_keys = list(all_results[first_model][first_snr].keys())
    
    header = ['Model', 'SNR'] + metric_keys
    
    print(f"[Data] Saving consolidated metrics to {filepath}...")
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for model_name, snr_dict in all_results.items():
            # Sort SNRs numerically for clean output
            sorted_snrs = sorted(snr_dict.keys(), key=lambda x: float(x))
            for snr in sorted_snrs:
                metrics = snr_dict[snr]
                row = [model_name, snr] + [metrics[k] for k in metric_keys]
                writer.writerow(row)
    print(f"[Success] Data saved to {filepath}")

if __name__ == "__main__":
    try:
        X, y, snr = load_rml2016_dataset(CONFIG['dataset_path'])
        
        # Normalize
        energy = np.sum(np.abs(X)**2, axis=(1, 2), keepdims=True)
        X = X / (np.sqrt(energy / (128*2)) + 1e-10)
        
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        snr_t = torch.tensor(snr, dtype=torch.float32)
        
        dataset = TensorDataset(X_t, y_t, snr_t)
        train_ds, test_ds = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
        
        # Updated DataLoader to fulfill --cpus-per-task=16
        train_loader = DataLoader(
            train_ds, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True, 
            num_workers=CONFIG['num_workers'], 
            pin_memory=CONFIG['pin_memory']
        )
        test_loader = DataLoader(
            test_ds, 
            batch_size=CONFIG['batch_size'], 
            shuffle=False, 
            num_workers=CONFIG['num_workers'], 
            pin_memory=CONFIG['pin_memory']
        )
        
        models_list = ['LSTM', 'GRU', 'CNN', 'MCLDNN', 'PETCGDNN']
        all_model_results = {}
        
        for name in models_list:
            res, _ = run_experiment(name, train_loader, test_loader)
            all_model_results[name] = res
            
        save_consolidated_csv(all_model_results)
        print("[Success] All results exported successfully.")
        
    except FileNotFoundError as e:
        print(f"[Error] {e}")