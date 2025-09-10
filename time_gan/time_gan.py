import time
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yfinance as yf
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from torch import nn, optim

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando dispositivo:", device)

TICKERS = ["AAPL", "MSFT", "GOOGL"]
START = "2018-01-01"
END = None
SEQUENCE_LEN = 24
FEATURES = ["Close"]
BATCH_SIZE = 64
HIDDEN_DIM = 24
NUM_LAYERS = 2
LEARNING_RATE = 1e-3

EPOCHS_PRE_AE = 30
EPOCHS_PRE_SUP = 30
EPOCHS_JOINT = 100

GENERATOR_STEPS = 1
DISCRIMINATOR_STEPS = 1
GAMMA = 1.0
ETA = 1e-4

def download_data(tickers, start, end, max_retries=3, sleep_time=2):
    """Download dados do Yahoo Finance com tratamento robusto de erros"""
    frames = []
    for ticker in tickers:
        for attempt in range(max_retries):
            try:
                print(f"Baixando {ticker} (tentativa {attempt+1}/{max_retries})...")
                df = yf.download(
                    ticker, start=start, end=end, auto_adjust=True, progress=False
                )

                if df.empty:
                    print(f"Aviso: Nenhum dado para {ticker}")
                    break

                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)

                df["Ticker"] = ticker
                df = df.dropna()

                if len(df) > 0:
                    frames.append(df)
                else:
                    print(f"Aviso: {ticker} sem dados válidos após limpeza")
                break

            except Exception as e:
                print(f"Erro ao baixar {ticker} (tentativa {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(sleep_time)
                else:
                    print(f"Falha definitiva ao baixar {ticker}, pulando...")

    if not frames:
        raise RuntimeError("Nenhum dado válido baixado. Verifique tickers e conexão.")

    data = pd.concat(frames, ignore_index=False)
    print(f"Total de registros baixados: {len(data)}")
    return data


def make_panel_sequences(df, features, seq_len):
    """Cria sequências com tratamento robusto de dados financeiros"""
    sequences = []

    for ticker, group in df.groupby("Ticker"):
        print(f"Processando {ticker}...")
        group = group.sort_index().copy()

        try:
            feat_data = group[features].copy()
        except KeyError as e:
            print(f"Feature não encontrada para {ticker}: {e}")
            continue

        for col in feat_data.columns:
            if col.lower() != "volume":
                feat_data[col] = feat_data[col].replace([0, np.inf, -np.inf], np.nan)
                feat_data[col] = feat_data[col].where(feat_data[col] > 0)

                feat_data[col] = np.log(feat_data[col]).diff()

        feat_data = feat_data.dropna()

        if len(feat_data) < seq_len:
            print(f"Dados insuficientes para {ticker}: {len(feat_data)} < {seq_len}")
            continue

        for col in feat_data.columns:
            q1 = feat_data[col].quantile(0.01)
            q99 = feat_data[col].quantile(0.99)
            feat_data[col] = feat_data[col].clip(q1, q99)

        arr = feat_data.values
        for i in range(len(arr) - seq_len + 1):
            sequence = arr[i:i+seq_len]
            if not (np.isnan(sequence).any() or np.isinf(sequence).any()):
                sequences.append(sequence)

    if not sequences:
        print("Aviso: Nenhuma sequência válida gerada")
        return np.empty((0, seq_len, len(features)), dtype=np.float32)

    sequences = np.array(sequences, dtype=np.float32)
    print(f"Sequências geradas: {len(sequences)}")
    return sequences


def train_val_split(seqs, val_ratio=0.1):
    """Split com validação"""
    n = len(seqs)
    if n == 0:
        return np.array([]), np.array([])

    idx = np.arange(n)
    np.random.shuffle(idx)
    cut = int(n * (1 - val_ratio))
    return seqs[idx[:cut]], seqs[idx[cut:]]

print("Baixando dados...")
try:
    raw = download_data(TICKERS, START, END)
except Exception as e:
    print(f"Erro no download: {e}")
    exit(1)

print("Gerando sequências...")
seqs = make_panel_sequences(raw, FEATURES, SEQUENCE_LEN)

if len(seqs) == 0:
    print("Erro: Nenhuma sequência gerada. Verifique os dados.")
    exit(1)

print(f"Shape das sequências: {seqs.shape}")

scaler = MinMaxScaler(feature_range=(0, 1))
N, T, F = seqs.shape

if np.isnan(seqs).any() or np.isinf(seqs).any():
    print("Removendo sequências com NaN/Inf...")
    valid_mask = ~(np.isnan(seqs).any(axis=(1,2)) | np.isinf(seqs).any(axis=(1,2)))
    seqs = seqs[valid_mask]
    N, T, F = seqs.shape
    print(f"Sequências válidas restantes: {N}")

if N == 0:
    print("Erro: Nenhuma sequência válida após limpeza.")
    exit(1)

seqs_2d = seqs.reshape(-1, F)
seqs_scaled_2d = scaler.fit_transform(seqs_2d)
seqs_scaled = seqs_scaled_2d.reshape(N, T, F)

train_seqs, val_seqs = train_val_split(seqs_scaled, val_ratio=0.1)

if len(train_seqs) == 0:
    print("Erro: Nenhuma sequência de treino após split.")
    exit(1)

print(f"Treino: {len(train_seqs)}, Validação: {len(val_seqs)}")


class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_ds = SeqDataset(train_seqs)
train_dl = DataLoader(train_ds, batch_size=min(BATCH_SIZE, len(train_seqs)),
                      shuffle=True, drop_last=True)

if len(val_seqs) > 0:
    val_ds = SeqDataset(val_seqs)
    val_dl = DataLoader(val_ds, batch_size=min(BATCH_SIZE, len(val_seqs)),
                        shuffle=False, drop_last=False)


class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h, _ = self.rnn(x)
        h = self.dropout(h)
        out = self.fc(h)
        return out

class Embedder(nn.Module):
    def __init__(self, x_dim, h_dim, num_layers):
        super().__init__()
        self.net = GRUModel(x_dim, h_dim, num_layers, h_dim)

    def forward(self, x):
        return torch.sigmoid(self.net(x))

class Recovery(nn.Module):
    def __init__(self, h_dim, x_dim, num_layers):
        super().__init__()
        self.net = GRUModel(h_dim, h_dim, num_layers, x_dim)

    def forward(self, h):
        return self.net(h)

class Generator(nn.Module):
    def __init__(self, z_dim, h_dim, num_layers):
        super().__init__()
        self.net = GRUModel(z_dim, h_dim, num_layers, h_dim)

    def forward(self, z):
        return torch.sigmoid(self.net(z))

class Supervisor(nn.Module):
    def __init__(self, h_dim, num_layers):
        super().__init__()
        self.net = GRUModel(h_dim, h_dim, num_layers, h_dim)

    def forward(self, h):
        return torch.sigmoid(self.net(h))

class Discriminator(nn.Module):
    def __init__(self, h_dim, num_layers):
        super().__init__()
        self.rnn = nn.GRU(h_dim, h_dim, num_layers=num_layers,
                         batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.fc = nn.Linear(h_dim, 1)

    def forward(self, h):
        r, _ = self.rnn(h)
        y = self.fc(r)
        return y.mean(dim=1)

X_dim = F
H_dim = HIDDEN_DIM
Z_dim = H_dim

E = Embedder(X_dim, H_dim, NUM_LAYERS).to(device)
R = Recovery(H_dim, X_dim, NUM_LAYERS).to(device)
G = Generator(Z_dim, H_dim, NUM_LAYERS).to(device)
S = Supervisor(H_dim, NUM_LAYERS).to(device)
D = Discriminator(H_dim, NUM_LAYERS).to(device)

opt_E = optim.Adam(E.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
opt_R = optim.Adam(R.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
opt_G = optim.Adam(list(G.parameters()) + list(S.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)
opt_D = optim.Adam(D.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)


mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()

def sample_Z(batch_size, seq_len, z_dim, device):
    return torch.randn(batch_size, seq_len, z_dim, device=device)

def pretrain_autoencoder(epochs):
    """Pré-treino do autoencoder com validação"""
    print("Iniciando pré-treino do autoencoder...")
    E.train(); R.train()

    for ep in range(1, epochs+1):
        ep_loss = 0.0
        n_batches = 0

        for x in train_dl:
            x = x.to(device)

            opt_E.zero_grad(); opt_R.zero_grad()

            h = E(x)
            x_recon = R(h)
            loss = mse(x_recon, x)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(E.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)

            opt_E.step(); opt_R.step()

            ep_loss += loss.item()
            n_batches += 1

        avg_loss = ep_loss / n_batches if n_batches > 0 else 0
        print(f"[AE] Epoch {ep}/{epochs} - loss: {avg_loss:.6f}")

def pretrain_supervisor(epochs):
    """Pré-treino do supervisor corrigido"""
    print("Iniciando pré-treino do supervisor...")
    E.eval(); G.train(); S.train()

    for ep in range(1, epochs+1):
        ep_loss = 0.0
        n_batches = 0

        for x in train_dl:
            x = x.to(device)
            batch_size, seq_len = x.size(0), x.size(1)

            opt_G.zero_grad()

            z = sample_Z(batch_size, seq_len, Z_dim, device)
            h_fake = G(z)
            h_supervised = S(h_fake)

            if seq_len > 1:
                loss_s = mse(h_supervised[:, :-1, :], h_fake[:, 1:, :])
            else:
                loss_s = mse(h_supervised, h_fake)

            loss_s.backward()

            torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0)

            opt_G.step()

            ep_loss += loss_s.item()
            n_batches += 1

        avg_loss = ep_loss / n_batches if n_batches > 0 else 0
        print(f"[SUP] Epoch {ep}/{epochs} - sup_loss: {avg_loss:.6f}")

def joint_training(epochs):
    """Treinamento conjunto corrigido"""
    print("Iniciando treinamento conjunto...")

    for ep in range(1, epochs+1):
        E.train(); R.train(); G.train(); S.train(); D.train()

        g_loss_ep, d_loss_ep, ae_loss_ep = 0.0, 0.0, 0.0
        n_batches = 0

        for x in train_dl:
            x = x.to(device)
            batch_size, seq_len = x.size(0), x.size(1)

            opt_E.zero_grad(); opt_R.zero_grad()
            h_real = E(x)
            x_recon = R(h_real)
            ae_loss = mse(x_recon, x)
            ae_loss.backward()
            torch.nn.utils.clip_grad_norm_(E.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(R.parameters(), 1.0)
            opt_E.step(); opt_R.step()
            ae_loss_ep += ae_loss.item()

            for _ in range(DISCRIMINATOR_STEPS):
                opt_D.zero_grad()

                with torch.no_grad():
                    h_real = E(x)

                z = sample_Z(batch_size, seq_len, Z_dim, device)
                h_fake = G(z)
                h_supervised = S(h_fake)

                d_real = D(h_real.detach())
                d_fake = D(h_supervised.detach())

                d_loss_real = bce(d_real, torch.ones_like(d_real))
                d_loss_fake = bce(d_fake, torch.zeros_like(d_fake))
                d_loss = d_loss_real + d_loss_fake

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0)
                opt_D.step()
                d_loss_ep += d_loss.item()

            for _ in range(GENERATOR_STEPS):
                opt_G.zero_grad()

                z = sample_Z(batch_size, seq_len, Z_dim, device)
                h_fake = G(z)
                h_supervised = S(h_fake)

                d_fake = D(h_supervised)
                g_loss_adv = bce(d_fake, torch.ones_like(d_fake))

                if seq_len > 1:
                    g_loss_sup = mse(h_supervised[:, :-1, :], h_fake[:, 1:, :])
                else:
                    g_loss_sup = torch.tensor(0.0, device=device)

                with torch.no_grad():
                    h_real = E(x)

                moment_loss = torch.tensor(0.0, device=device)
                if ETA > 0:
                    mean_real = h_real.mean(dim=(0, 1))
                    mean_fake = h_supervised.mean(dim=(0, 1))
                    var_real = h_real.var(dim=(0, 1))
                    var_fake = h_supervised.var(dim=(0, 1))

                    moment_loss = (mean_real - mean_fake).pow(2).mean() + \
                                 (var_real - var_fake).pow(2).mean()

                g_loss = g_loss_adv + GAMMA * g_loss_sup + ETA * moment_loss

                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(S.parameters(), 1.0)
                opt_G.step()
                g_loss_ep += g_loss.item()

            n_batches += 1

        if n_batches > 0:
            print(f"[JOINT] Epoch {ep}/{epochs} | "
                  f"AE:{ae_loss_ep/n_batches:.5f} | "
                  f"D:{d_loss_ep/(n_batches * DISCRIMINATOR_STEPS):.5f} | "
                  f"G:{g_loss_ep/(n_batches * GENERATOR_STEPS):.5f}")


print("\nEtapa 1/3: Pré-treino Autoencoder...")
pretrain_autoencoder(EPOCHS_PRE_AE)

print("\nEtapa 2/3: Pré-treino Supervisor...")
pretrain_supervisor(EPOCHS_PRE_SUP)

print("\nEtapa 3/3: Treino Conjunto...")
joint_training(EPOCHS_JOINT)

def generate_synthetic(n_samples, seq_len):
    """Gerar amostras sintéticas"""
    E.eval(); R.eval(); G.eval(); S.eval()

    with torch.no_grad():
        z = sample_Z(n_samples, seq_len, Z_dim, device)
        h_fake = G(z)
        h_supervised = S(h_fake)
        x_fake = R(h_supervised)
        return x_fake.cpu().numpy()

print("\nGerando séries sintéticas...")
n_synthetic = min(256, len(train_seqs))
synth_scaled = generate_synthetic(n_synthetic, SEQUENCE_LEN)

synth_2d = synth_scaled.reshape(-1, F)
synth_original = scaler.inverse_transform(synth_2d).reshape(n_synthetic, SEQUENCE_LEN, F)

print(f"Séries sintéticas geradas: {synth_original.shape}")

real_scaled = train_seqs[:n_synthetic]
real_2d = real_scaled.reshape(-1, F)
real_original = scaler.inverse_transform(real_2d).reshape(n_synthetic, SEQUENCE_LEN, F)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for i in range(min(5, len(real_original))):
    plt.plot(real_original[i, :, 0], alpha=0.7, label=f'Real {i+1}')
plt.title('Séries Temporais Reais')
plt.xlabel('Tempo')
plt.ylabel('Log Returns')
plt.legend()

plt.subplot(1, 2, 2)
for i in range(min(5, len(synth_original))):
    plt.plot(synth_original[i, :, 0], alpha=0.7, label=f'Sintético {i+1}')
plt.title('Séries Temporais Sintéticas')
plt.xlabel('Tempo')
plt.ylabel('Log Returns')
plt.legend()

plt.tight_layout()
plt.show()

real_vals = real_original[:, :, 0].flatten()
synth_vals = synth_original[:, :, 0].flatten()

print("\n" + "="*50)
print("ESTATÍSTICAS COMPARATIVAS")
print("="*50)
print(f"Real  : mean={real_vals.mean():.6f} std={real_vals.std():.6f}")
print(f"Synth : mean={synth_vals.mean():.6f} std={synth_vals.std():.6f}")
print(f"Real  : min={real_vals.min():.6f} max={real_vals.max():.6f}")
print(f"Synth : min={synth_vals.min():.6f} max={synth_vals.max():.6f}")



# Assumindo que os modelos já estão treinados (E, R, G, S) e o scaler está disponível
# Se não estiver, descomente e ajuste as linhas abaixo:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SEQ_LEN = 24
# Z_dim = 24
# F = 1  # número de features

def sample_Z(batch_size, seq_len, z_dim, device, seed=None):
    """Gerar ruído aleatório para o gerador com controle de seed"""
    if seed is not None:
        torch.manual_seed(seed)
    # Usar diferentes tipos de distribuições para mais diversidade
    z = torch.randn(batch_size, seq_len, z_dim, device=device)
    return z

def generate_synthetic_batch(n_samples, seq_len, batch_size=64, add_noise=True, temperature=1.0):
    """
    Gerar múltiplas séries sintéticas com controles para diversidade

    Args:
        n_samples: número de séries a gerar
        seq_len: comprimento de cada série
        batch_size: tamanho do batch
        add_noise: adicionar ruído extra para diversidade
        temperature: controlar a "temperatura" da geração (maior = mais diverso)
    """
    print(f"Gerando {n_samples} séries sintéticas...")
    print(f"  Temperature: {temperature}")
    print(f"  Add noise: {add_noise}")

    # Colocar modelos em modo de avaliação mas com dropout ativo se necessário
    E.eval(); R.eval(); G.eval(); S.eval()

    # Para garantir diversidade, podemos ativar dropout mesmo em eval
    if hasattr(G, 'training'):
        G.train()  # Manter dropout ativo para mais diversidade
    if hasattr(S, 'training'):
        S.train()

    all_synthetic = []

    # Usar seeds diferentes para cada batch
    np.random.seed(None)  # Reset seed
    torch.manual_seed(np.random.randint(0, 10000))

    for i in range(0, n_samples, batch_size):
        current_batch_size = min(batch_size, n_samples - i)

        with torch.no_grad():
            # Gerar ruído com seed único para cada batch
            batch_seed = np.random.randint(0, 100000)
            z = sample_Z(current_batch_size, seq_len, Z_dim, device, seed=batch_seed)

            # Aplicar temperature scaling ao ruído
            z = z * temperature

            # Adicionar ruído extra se solicitado
            if add_noise:
                noise_scale = 0.1
                extra_noise = torch.randn_like(z) * noise_scale
                z = z + extra_noise

            # Passar pelo gerador
            h_fake = G(z)

            # Adicionar pequeno ruído ao estado oculto para mais diversidade
            if add_noise:
                h_noise = torch.randn_like(h_fake) * 0.05
                h_fake = h_fake + h_noise

            # Passar pelo supervisor
            h_supervised = S(h_fake)

            # Reconstruir séries temporais
            x_fake = R(h_supervised)

            # Adicionar ruído final muito pequeno (opcional)
            if add_noise:
                final_noise = torch.randn_like(x_fake) * 0.01
                x_fake = x_fake + final_noise

            all_synthetic.append(x_fake.cpu().numpy())

            if (i // batch_size + 1) % 10 == 0:
                print(f"  Batch {i//batch_size + 1}/{(n_samples-1)//batch_size + 1} concluído")

    # Concatenar todos os batches
    synthetic_data = np.concatenate(all_synthetic, axis=0)

    # Verificar diversidade antes da desnormalização
    print(f"  Variância dos dados sintéticos (normalizado): {synthetic_data.var():.6f}")
    print(f"  Range dos dados sintéticos: {synthetic_data.min():.6f} a {synthetic_data.max():.6f}")

    # Inverter normalização
    synth_2d = synthetic_data.reshape(-1, F)
    synth_original = scaler.inverse_transform(synth_2d).reshape(n_samples, seq_len, F)

    # Verificar diversidade após desnormalização
    print(f"  Variância após desnormalização: {synth_original.var():.6f}")
    print(f"  Range após desnormalização: {synth_original.min():.6f} a {synth_original.max():.6f}")

    return synth_original

def check_model_diversity():
    """
    Verificar se os modelos estão gerando diversidade
    """
    print("\n" + "="*50)
    print("VERIFICAÇÃO DE DIVERSIDADE DOS MODELOS")
    print("="*50)

    # Testar com o mesmo ruído
    print("1. Testando com ruído idêntico...")
    z_test = torch.randn(5, SEQUENCE_LEN, Z_dim, device=device)
    z_repeated = z_test.repeat(1, 1, 1)  # Mesmo ruído

    with torch.no_grad():
        G.eval(); S.eval(); R.eval()
        h1 = G(z_repeated)
        h2 = S(h1)
        x1 = R(h2)

        print(f"  Variância com ruído idêntico: {x1.var().item():.6f}")
        print(f"  Diferença máxima entre séries: {(x1.max() - x1.min()).item():.6f}")

    # Testar com ruído diferente
    print("\n2. Testando com ruído diferente...")
    z_different = torch.randn(5, SEQUENCE_LEN, Z_dim, device=device)

    with torch.no_grad():
        h1 = G(z_different)
        h2 = S(h1)
        x2 = R(h2)

        print(f"  Variância com ruído diferente: {x2.var().item():.6f}")
        print(f"  Diferença máxima entre séries: {(x2.max() - x2.min()).item():.6f}")

    # Comparar as duas
    print("\n3. Comparação:")
    with torch.no_grad():
        diff = torch.abs(x1 - x2).mean()
        print(f"  Diferença média entre os dois casos: {diff.item():.6f}")

    print("="*50)

def plot_individual_series(synthetic_data, n_series=10, feature_idx=0, title_prefix="Série"):
    """
    Plot de séries individuais
    """
    plt.figure(figsize=(15, 8))

    n_cols = 5
    n_rows = (n_series + n_cols - 1) // n_cols

    for i in range(min(n_series, len(synthetic_data))):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(synthetic_data[i, :, feature_idx], linewidth=2)
        plt.title(f'{title_prefix} {i+1}')
        plt.xlabel('Tempo')
        plt.ylabel('Valor')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_overlay_series(synthetic_data, n_series=20, feature_idx=0, alpha=0.6):
    """
    Plot sobreposto de múltiplas séries
    """
    plt.figure(figsize=(12, 6))

    for i in range(min(n_series, len(synthetic_data))):
        plt.plot(synthetic_data[i, :, feature_idx], alpha=alpha, linewidth=1.5)

    plt.title(f'Sobreposição de {min(n_series, len(synthetic_data))} Séries Sintéticas')
    plt.xlabel('Tempo')
    plt.ylabel('Valor')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_statistical_analysis(synthetic_data, feature_idx=0):
    """
    Análise estatística das séries geradas
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Preparar dados
    all_values = synthetic_data[:, :, feature_idx].flatten()
    series_means = synthetic_data[:, :, feature_idx].mean(axis=1)
    series_stds = synthetic_data[:, :, feature_idx].std(axis=1)
    series_mins = synthetic_data[:, :, feature_idx].min(axis=1)
    series_maxs = synthetic_data[:, :, feature_idx].max(axis=1)

    # 1. Distribuição de todos os valores
    axes[0, 0].hist(all_values, bins=50, alpha=0.7, density=True, color='skyblue')
    axes[0, 0].set_title('Distribuição de Todos os Valores')
    axes[0, 0].set_xlabel('Valor')
    axes[0, 0].set_ylabel('Densidade')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Q-Q plot para normalidade
    stats.probplot(all_values, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normalidade)')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Distribuição das médias das séries
    axes[0, 2].hist(series_means, bins=30, alpha=0.7, color='lightcoral')
    axes[0, 2].set_title('Distribuição das Médias por Série')
    axes[0, 2].set_xlabel('Média')
    axes[0, 2].set_ylabel('Frequência')
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Distribuição dos desvios padrão
    axes[1, 0].hist(series_stds, bins=30, alpha=0.7, color='lightgreen')
    axes[1, 0].set_title('Distribuição dos Desvios Padrão')
    axes[1, 0].set_xlabel('Desvio Padrão')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Box plot dos valores min/max
    box_data = [series_mins, series_maxs]
    axes[1, 1].boxplot(box_data, labels=['Mínimos', 'Máximos'])
    axes[1, 1].set_title('Box Plot: Valores Extremos por Série')
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Evolução temporal média
    mean_evolution = synthetic_data[:, :, feature_idx].mean(axis=0)
    std_evolution = synthetic_data[:, :, feature_idx].std(axis=0)

    x_axis = range(len(mean_evolution))
    axes[1, 2].plot(x_axis, mean_evolution, 'b-', linewidth=2, label='Média')
    axes[1, 2].fill_between(x_axis,
                           mean_evolution - std_evolution,
                           mean_evolution + std_evolution,
                           alpha=0.3, label='±1 Desvio Padrão')
    axes[1, 2].set_title('Evolução Temporal Média')
    axes[1, 2].set_xlabel('Tempo')
    axes[1, 2].set_ylabel('Valor')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_autocorrelation_analysis(synthetic_data, feature_idx=0, max_lags=10):
    """
    Análise de autocorrelação das séries
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Calcular autocorrelações para diferentes lags
    n_series = min(50, len(synthetic_data))  # Usar no máximo 50 séries para eficiência
    autocorrs = []

    for i in range(n_series):
        series = synthetic_data[i, :, feature_idx]
        series_autocorr = []
        for lag in range(1, max_lags + 1):
            if len(series) > lag:
                autocorr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
                series_autocorr.append(autocorr if not np.isnan(autocorr) else 0)
            else:
                series_autocorr.append(0)
        autocorrs.append(series_autocorr)

    autocorrs = np.array(autocorrs)

    # 1. Heatmap de autocorrelações
    sns.heatmap(autocorrs[:20], # Mostrar apenas 20 séries
                annot=False,
                xticklabels=range(1, max_lags + 1),
                yticklabels=[f'Série {i+1}' for i in range(min(20, n_series))],
                cmap='coolwarm',
                center=0,
                ax=axes[0, 0])
    axes[0, 0].set_title('Autocorrelação por Série')
    axes[0, 0].set_xlabel('Lag')

    # 2. Autocorrelação média
    mean_autocorr = np.nanmean(autocorrs, axis=0)
    std_autocorr = np.nanstd(autocorrs, axis=0)

    lags = range(1, max_lags + 1)
    axes[0, 1].bar(lags, mean_autocorr, yerr=std_autocorr,
                   alpha=0.7, capsize=5, color='steelblue')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('Autocorrelação Média por Lag')
    axes[0, 1].set_xlabel('Lag')
    axes[0, 1].set_ylabel('Autocorrelação')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Distribuição das autocorrelações lag-1
    axes[1, 0].hist(autocorrs[:, 0], bins=20, alpha=0.7, color='orange')
    axes[1, 0].set_title('Distribuição Autocorrelação Lag-1')
    axes[1, 0].set_xlabel('Autocorrelação')
    axes[1, 0].set_ylabel('Frequência')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Exemplo de algumas séries individuais
    for i in range(min(5, n_series)):
        axes[1, 1].plot(lags, autocorrs[i], alpha=0.6, marker='o', linewidth=1)
    axes[1, 1].set_title('Autocorrelação: Exemplos Individuais')
    axes[1, 1].set_xlabel('Lag')
    axes[1, 1].set_ylabel('Autocorrelação')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def compare_with_original(synthetic_data, original_data, feature_idx=0, n_comparison=10):
    """
    Comparar séries sintéticas com originais
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Séries lado a lado
    for i in range(min(n_comparison, len(synthetic_data), len(original_data))):
        axes[0, 0].plot(original_data[i, :, feature_idx],
                       alpha=0.7, linewidth=1, color='blue')
        axes[0, 1].plot(synthetic_data[i, :, feature_idx],
                       alpha=0.7, linewidth=1, color='red')

    axes[0, 0].set_title(f'Séries Originais (n={min(n_comparison, len(original_data))})')
    axes[0, 0].set_xlabel('Tempo')
    axes[0, 0].set_ylabel('Valor')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title(f'Séries Sintéticas (n={min(n_comparison, len(synthetic_data))})')
    axes[0, 1].set_xlabel('Tempo')
    axes[0, 1].set_ylabel('Valor')
    axes[0, 1].grid(True, alpha=0.3)

    # 2. Comparação de distribuições
    orig_values = original_data[:, :, feature_idx].flatten()
    synth_values = synthetic_data[:, :, feature_idx].flatten()

    axes[1, 0].hist(orig_values, bins=50, alpha=0.6, label='Original',
                    density=True, color='blue')
    axes[1, 0].hist(synth_values, bins=50, alpha=0.6, label='Sintético',
                    density=True, color='red')
    axes[1, 0].set_title('Comparação de Distribuições')
    axes[1, 0].set_xlabel('Valor')
    axes[1, 0].set_ylabel('Densidade')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 3. Estatísticas comparativas
    orig_stats = [orig_values.mean(), orig_values.std(),
                 np.percentile(orig_values, 25), np.percentile(orig_values, 75)]
    synth_stats = [synth_values.mean(), synth_values.std(),
                  np.percentile(synth_values, 25), np.percentile(synth_values, 75)]

    x_labels = ['Média', 'Desvio Padrão', 'Q1', 'Q3']
    x_pos = np.arange(len(x_labels))

    width = 0.35
    axes[1, 1].bar(x_pos - width/2, orig_stats, width,
                   label='Original', color='blue', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, synth_stats, width,
                   label='Sintético', color='red', alpha=0.7)

    axes[1, 1].set_title('Estatísticas Comparativas')
    axes[1, 1].set_xlabel('Métrica')
    axes[1, 1].set_ylabel('Valor')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(x_labels)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def print_generation_summary(synthetic_data):
    """
    Imprimir resumo da geração
    """
    n_series, seq_len, n_features = synthetic_data.shape

    print("\n" + "="*60)
    print("RESUMO DA GERAÇÃO DE SÉRIES SINTÉTICAS")
    print("="*60)
    print(f"Número de séries geradas: {n_series}")
    print(f"Comprimento de cada série: {seq_len}")
    print(f"Número de features: {n_features}")

    for feat_idx in range(n_features):
        values = synthetic_data[:, :, feat_idx].flatten()
        print(f"\nFeature {feat_idx}:")
        print(f"  Média: {values.mean():.6f}")
        print(f"  Desvio Padrão: {values.std():.6f}")
        print(f"  Mínimo: {values.min():.6f}")
        print(f"  Máximo: {values.max():.6f}")
        print(f"  Mediana: {np.median(values):.6f}")

        # Teste de normalidade
        _, p_value = stats.normaltest(values)
        print(f"  Teste normalidade (p-value): {p_value:.6f}")

    print("="*60)

# ==========================
# EXECUÇÃO PRINCIPAL COM DIAGNÓSTICO
# ==========================

if __name__ == "__main__":
    # Configurações
    N_SYNTHETIC_SERIES = 100  # Número de séries a gerar
    FEATURE_IDX = 0  # Índice da feature a analisar (0 para 'Close')

    print("Iniciando geração de séries temporais sintéticas...")

    # DIAGNÓSTICO: Verificar se os modelos geram diversidade
    check_model_diversity()

    # 1. Gerar séries sintéticas com diferentes configurações
    print("\n" + "="*50)
    print("TESTANDO DIFERENTES CONFIGURAÇÕES")
    print("="*50)

    configurations = [
        {"add_noise": False, "temperature": 1.0, "name": "Padrão"},
        {"add_noise": True, "temperature": 1.0, "name": "Com ruído extra"},
        {"add_noise": True, "temperature": 1.5, "name": "Temperature alta + ruído"},
        {"add_noise": True, "temperature": 2.0, "name": "Temperature muito alta"},
    ]

    best_synthetic = None
    best_diversity = 0

    for config in configurations:
        print(f"\nTestando configuração: {config['name']}")
        synthetic_test = generate_synthetic_batch(
            20, SEQUENCE_LEN,
            add_noise=config["add_noise"],
            temperature=config["temperature"]
        )

        # Calcular diversidade (variância entre séries)
        diversity = np.var([series.var() for series in synthetic_test[:, :, FEATURE_IDX]])
        print(f"  Diversidade (variância das variâncias): {diversity:.6f}")

        # Verificar se séries são idênticas
        first_series = synthetic_test[0, :, FEATURE_IDX]
        identical_count = 0
        for i in range(1, min(10, len(synthetic_test))):
            if np.allclose(first_series, synthetic_test[i, :, FEATURE_IDX], rtol=1e-3):
                identical_count += 1

        print(f"  Séries ~idênticas às primeiras: {identical_count}/9")

        if diversity > best_diversity:
            best_diversity = diversity
            best_synthetic = synthetic_test
            print(f"  ✓ Melhor configuração até agora!")

    # 2. Gerar dataset final com a melhor configuração
    if best_synthetic is not None:
        print(f"\nUsando melhor configuração para gerar {N_SYNTHETIC_SERIES} séries...")
        synthetic_series = generate_synthetic_batch(
            N_SYNTHETIC_SERIES, SEQUENCE_LEN,
            add_noise=True,
            temperature=1.5  # Usar configuração que mostrou melhor diversidade
        )
    else:
        print("Usando configuração padrão...")
        synthetic_series = generate_synthetic_batch(N_SYNTHETIC_SERIES, SEQUENCE_LEN)

    # 3. Verificação final de diversidade
    print("\n" + "="*50)
    print("VERIFICAÇÃO FINAL DE DIVERSIDADE")
    print("="*50)

    # Calcular estatísticas de diversidade
    series_means = [series[:, FEATURE_IDX].mean() for series in synthetic_series]
    series_stds = [series[:, FEATURE_IDX].std() for series in synthetic_series]

    print(f"Diversidade das médias: {np.std(series_means):.6f}")
    print(f"Diversidade dos desvios padrão: {np.std(series_stds):.6f}")

    # Verificar correlações entre séries (devem ser baixas)
    correlations = []
    for i in range(min(10, len(synthetic_series))):
        for j in range(i+1, min(10, len(synthetic_series))):
            corr = np.corrcoef(synthetic_series[i, :, FEATURE_IDX],
                             synthetic_series[j, :, FEATURE_IDX])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    if correlations:
        print(f"Correlação média entre séries: {np.mean(correlations):.6f}")
        print(f"Correlação máxima: {np.max(correlations):.6f}")

        if np.mean(correlations) > 0.8:
            print("⚠️  AVISO: Correlações altas - séries podem ser muito similares")
        elif np.mean(correlations) < 0.3:
            print("✓ Boa diversidade - correlações baixas entre séries")

    # 4. Resumo da geração
    print_generation_summary(synthetic_series)

    # 5. Visualizações (apenas se há diversidade suficiente)
    if np.std(series_means) > 1e-6:  # Threshold mínimo de diversidade
        print("\nGerando visualizações...")

        # Séries individuais
        print("  - Séries individuais...")
        plot_individual_series(synthetic_series, n_series=12, feature_idx=FEATURE_IDX)

        # Séries sobrepostas
        print("  - Sobreposição de séries...")
        plot_overlay_series(synthetic_series, n_series=50, feature_idx=FEATURE_IDX)

        # Análise estatística
        print("  - Análise estatística...")
        plot_statistical_analysis(synthetic_series, feature_idx=FEATURE_IDX)

        # Análise de autocorrelação
        print("  - Análise de autocorrelação...")
        plot_autocorrelation_analysis(synthetic_series, feature_idx=FEATURE_IDX)

        # Comparação com dados originais (se disponível)
        if 'train_seqs' in globals() and len(train_seqs) > 0:
            print("  - Comparação com dados originais...")
            # Inverter normalização dos dados originais
            real_2d = train_seqs.reshape(-1, F)
            real_original = scaler.inverse_transform(real_2d).reshape(len(train_seqs), SEQUENCE_LEN, F)
            compare_with_original(synthetic_series, real_original, feature_idx=FEATURE_IDX)

        print("\nVisualização concluída!")
    else:
        print("\n⚠️  Séries são muito similares - pulando visualizações")
        print("Possíveis soluções:")
        print("1. Re-treinar o modelo com mais épocas")
        print("2. Aumentar a dimensão do ruído (Z_dim)")
        print("3. Modificar a arquitetura do gerador")
        print("4. Ajustar hiperparâmetros de treinamento")

    print("\nDicas para melhorar diversidade:")
    print("1. Verificar se o treinamento convergiu adequadamente")
    print("2. Aumentar temperatura na geração")
    print("3. Adicionar dropout nos modelos durante geração")
    print("4. Usar diferentes seeds para cada série")
    print("5. Implementar techniques como nucleus sampling")
