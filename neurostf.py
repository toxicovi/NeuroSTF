import os
import random
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ----------------------------
# Configuration and Setup
# ----------------------------
class Config:
    seq_len = 96
    pred_len = 24
    batch_size = 32
    d_model = 128
    n_heads = 4
    num_layers = 4
    expansion_factor = 2
    dropout = 0.1
    attn_dropout = 0.1
    conv_kernel = 5
    epochs = 40
    lr = 3e-4
    weight_decay = 1e-6
    patience = 7
    grad_clip = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42

    # Ablation toggles
    ablate_attention = False
    ablate_conv = False
    ablate_pruning = False

    results_dir = "results"
    logs_dir = "logs"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(Config.seed)

# ----------------------------
# Data Loading
# ----------------------------
class ETTHourly(Dataset):
    def __init__(self, file_path, split='train'):
        df = pd.read_csv(file_path)
        self.data = df.iloc[:, 1:].values.astype(np.float32)
        self.mean = np.nanmean(self.data, axis=0)
        self.std = np.nanstd(self.data, axis=0) + 1e-8
        self.data = (self.data - self.mean) / self.std
        L = len(self.data)
        if split == 'train':
            self.data = self.data[:int(0.6 * L)]
        elif split == 'val':
            self.data = self.data[int(0.6 * L):int(0.8 * L)]
        else:
            self.data = self.data[int(0.8 * L):]

    def __len__(self):
        return len(self.data) - Config.seq_len - Config.pred_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + Config.seq_len]
        y = self.data[idx + Config.seq_len:idx + Config.seq_len + Config.pred_len]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# ----------------------------
# Positional Encoding Module
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        position = torch.arange(0, Config.seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, Config.seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ----------------------------
# Neuroplastic Sparse Attention
# ----------------------------
class NeuroplasticSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(Config.attn_dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)

        if self.training and not Config.ablate_pruning:
            threshold = 0.01
            growth_rate = 0.05
            sparse_mask = (attn > threshold).float()
            regrow = (torch.rand_like(sparse_mask) < growth_rate).float()
            attn = attn * torch.max(sparse_mask, regrow)

        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)

# ----------------------------
# Temporal Mixer Block
# ----------------------------
class TemporalMixerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = NeuroplasticSparseAttention(d_model, Config.n_heads)

        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=Config.conv_kernel, padding=Config.conv_kernel // 2),
            nn.GLU(dim=1),
            nn.BatchNorm1d(d_model)
        )

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * Config.expansion_factor),
            nn.GELU(),
            nn.Linear(d_model * Config.expansion_factor, d_model)
        )
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, x):
        if not Config.ablate_attention:
            x = x + self.dropout(self.attn(self.attn_norm(x)))
        if not Config.ablate_conv:
            x_t = self.conv_norm(x).transpose(1, 2)
            x = x + self.dropout(self.conv(x_t).transpose(1, 2))
        x = x + self.dropout(self.mlp(self.mlp_norm(x)))
        return x

# ----------------------------
# NeuroSTF Model
# ----------------------------
class NeuroSTF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, Config.d_model)
        self.pos_enc = PositionalEncoding(Config.d_model)
        self.dropout = nn.Dropout(Config.dropout)
        self.blocks = nn.ModuleList([TemporalMixerBlock(Config.d_model) for _ in range(Config.num_layers)])
        self.predictor = nn.Sequential(
            nn.LayerNorm(Config.d_model),
            nn.Linear(Config.d_model, Config.pred_len * input_dim)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        # Use last time-step representation for prediction
        return self.predictor(x[:, -1, :]).view(x.size(0), Config.pred_len, -1)

# ----------------------------
# Benchmark Model Placeholders
# Replace these with official implementations in production
# ----------------------------
class SimpleFEDformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(Config.seq_len * input_dim, Config.pred_len * input_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        return self.fc(x).view(x.size(0), Config.pred_len, -1)

class SimpleTimesNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(Config.pred_len)
        self.fc = nn.Linear(64, input_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # B,C,T
        x = F.relu(self.conv1(x))
        x = self.pool(x)  # B, C, pred_len
        x = x.transpose(1, 2)  # B, pred_len, C
        return self.fc(x)

class SimpleAutoformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(Config.seq_len * input_dim, 256)
        self.fc2 = nn.Linear(256, Config.pred_len * input_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).view(x.size(0), Config.pred_len, -1)

# ----------------------------
# Trainer
# ----------------------------
class Trainer:
    def __init__(self, model, train_loader, val_loader, model_name, dataset_name):
        self.model = model.to(Config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=Config.patience//2)
        self.criterion = nn.HuberLoss()
        self.model_name = model_name
        self.dataset_name = dataset_name

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in tqdm(self.train_loader, desc=f"Training {self.model_name} on {self.dataset_name}"):
            x, y = x.to(Config.device), y.to(Config.device)
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), Config.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(Config.device), y.to(Config.device)
                pred = self.model(x)
                total_loss += self.criterion(pred, y).item()
        return total_loss / len(self.val_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(Config.device), y.to(Config.device)
                pred = self.model(x)
                preds.append(pred.cpu().numpy())
                trues.append(y.cpu().numpy())
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        mse = mean_squared_error(trues, preds)
        mae = mean_absolute_error(trues, preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((trues - preds) / (trues + 1e-8))) * 100
        da = np.mean(np.sign(trues[1:] - trues[:-1]) == np.sign(preds[1:] - preds[:-1]))
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'DA': da}

    def save_model(self, epoch):
        os.makedirs(Config.results_dir, exist_ok=True)
        path = f"{Config.results_dir}/best_{self.model_name}_{self.dataset_name}.pth"
        torch.save(self.model.state_dict(), path)

    def load_model(self):
        path = f"{Config.results_dir}/best_{self.model_name}_{self.dataset_name}.pth"
        self.model.load_state_dict(torch.load(path, map_location=Config.device))

# ----------------------------
# Visualization
# ----------------------------
def plot_predictions(true_vals, pred_vals, title, dataset_name, model_name, sample_idx=0):
    os.makedirs(Config.results_dir, exist_ok=True)
    plt.figure(figsize=(10,5))
    plt.plot(true_vals[sample_idx, :, 0], label='True')
    plt.plot(pred_vals[sample_idx, :, 0], label='Predicted')
    plt.title(f"{title} - {dataset_name} - {model_name}")
    plt.legend()
    plt.savefig(f"{Config.results_dir}/{model_name}_{dataset_name}_sample{sample_idx}.png")
    plt.close()

# ----------------------------
# Run Experiment
# ----------------------------
def run_experiment(model_cls, dataset_path, model_name, dataset_name):
    train_set = ETTHourly(dataset_path, 'train')
    val_set = ETTHourly(dataset_path, 'val')
    test_set = ETTHourly(dataset_path, 'test')

    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size, shuffle=False)

    model = model_cls(train_set.data.shape[1])
    trainer = Trainer(model, train_loader, val_loader, model_name, dataset_name)

    best_val_loss = float('inf')
    no_improve = 0
    for epoch in range(Config.epochs):
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()
        trainer.scheduler.step(val_loss)
        print(f"Epoch {epoch+1}/{Config.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            trainer.save_model(epoch)
        else:
            no_improve += 1
            if no_improve >= Config.patience:
                print("Early stopping triggered.")
                break

    trainer.load_model()
    metrics = trainer.evaluate(test_loader)
    print(f"Test metrics for {model_name} on {dataset_name}: {metrics}")

    # Plot prediction of first sample
    with torch.no_grad():
        x_sample, y_sample = test_set[0]
        pred_sample = model(x_sample.unsqueeze(0).to(Config.device)).cpu().numpy()
    plot_predictions(y_sample.numpy(), pred_sample, "Prediction vs True", dataset_name, model_name)

    return metrics

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    datasets = ["ETTh1.csv", "ETTh2.csv"]
    model_experiments = [
        ("NeuroSTF", NeuroSTF),
        ("FEDformer", SimpleFEDformer),
        ("TimesNet", SimpleTimesNet),
        ("Autoformer", SimpleAutoformer)
    ]

    all_results = {}
    for dataset in datasets:
        print(f"\n{'='*60}\nRunning experiments on {dataset}\n{'='*60}")
        all_results[dataset] = {}
        for model_name, model_cls in model_experiments:
            print(f"\n--- {model_name} on {dataset} ---")
            Config.ablate_attention = False
            Config.ablate_conv = False
            Config.ablate_pruning = False
            results = run_experiment(model_cls, dataset, model_name, dataset.split('.')[0])
            all_results[dataset][model_name] = results

    print("\n\n====== SUMMARY ======")
    for dataset, models in all_results.items():
        print(f"\nDataset: {dataset}")
        for model_name, metrics in models.items():
            print(f"  {model_name}: MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%, DA={metrics['DA']:.4f}")

