import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import warnings

warnings.filterwarnings('ignore')

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")


# 1. Coleta de dados do yfinance
def carregar_dados(ticker='AAPL', periodo='2y', intervalo='1d'):
    """Baixa dados históricos do yfinance"""
    print(f"Baixando dados de {ticker}...")
    dados = yf.download(ticker, period=periodo, interval=intervalo, progress=False)
    return dados['Close'].values.reshape(-1, 1)


# 2. Preparação dos dados com Data Augmentation
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=60, pred_len=5, augment=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.augment = augment
        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len].copy()
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len].copy()

        # Data augmentation durante treino
        if self.augment:
            # Adicionar ruído gaussiano
            if np.random.rand() > 0.5:
                noise = np.random.normal(0, 0.02, x.shape)
                x = x + noise

            # Scaling aleatório
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.95, 1.05)
                x = x * scale

        return torch.FloatTensor(x), torch.FloatTensor(y)


# 3. Componentes da arquitetura TimeGPT com regularização melhorada
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeGPT(nn.Module):
    def __init__(self, input_dim=1, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.2, pred_len=5):
        super().__init__()
        self.d_model = d_model
        self.pred_len = pred_len

        # Embedding da entrada com dropout
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN para melhor estabilidade
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # Cabeça de predição com mais regularização
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(dim_feedforward, dim_feedforward // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, pred_len * input_dim)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.input_embedding(x)
        x = self.pos_encoding(x)

        # Passar pelo transformer
        x = self.transformer_encoder(x)

        # Usar pooling ao invés de apenas o último estado
        x = torch.mean(x[:, -10:, :], dim=1)  # Average dos últimos 10 steps

        # Gerar predições
        output = self.output_proj(x)
        output = output.reshape(-1, self.pred_len, 1)

        return output


# 4. Treinamento com Early Stopping e Weight Decay
class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def treinar_modelo(model, train_loader, val_loader, epochs=100, lr=0.01):
    criterion = nn.MSELoss()
    # Adicionar weight decay para regularização L2
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=15)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    print("\nIniciando treinamento...")
    for epoch in range(epochs):
        # Treino
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validação
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                loss = criterion(output, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        scheduler.step()

        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # Early stopping
        # early_stopping(val_loss)
        # if early_stopping.early_stop:
        #     print(f"\nEarly stopping na época {epoch+1}")
        #     break

    # Carregar melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses


# 5. Geração de novas séries
def gerar_serie_futura(model, dataset, num_steps=50):
    """Gera uma série futura autoregressivamente"""
    model.eval()

    # Pegar última sequência do dataset
    ultimo_idx = len(dataset) - 1
    x, _ = dataset[ultimo_idx]
    sequencia = x.numpy().copy()

    predicoes = []

    with torch.no_grad():
        for _ in range(num_steps):
            # Preparar input
            x_input = torch.FloatTensor(sequencia[-dataset.seq_len:]).unsqueeze(0).to(device)

            # Predizer próximo passo
            pred = model(x_input)
            proximo_valor = pred[0, 0].cpu().numpy()  # Apenas t+1

            # Adicionar à sequência
            predicoes.append(proximo_valor)
            sequencia = np.vstack([sequencia, proximo_valor.reshape(1, -1)])

    return np.array(predicoes)


# 6. Avaliação
def avaliar_modelo(model, dataset, num_predictions=100):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for i in range(min(num_predictions, len(dataset))):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x)
            predictions.append(pred.cpu().numpy()[0])
            actuals.append(y.numpy())

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    return predictions, actuals


# 7. Visualizações completas
def plotar_resultados_completos(predictions, actuals, serie_futura, dados_originais,
                                train_losses, val_losses, dataset):
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Loss durante treinamento
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train_losses, label='Train Loss', linewidth=2)
    ax1.plot(val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('Curva de Aprendizado', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # 2. Predições vs Real (t+1)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(actuals[:, 0], label='Real', alpha=0.7, linewidth=2)
    ax2.plot(predictions[:, 0], label='Predito', alpha=0.7, linewidth=2)
    ax2.set_xlabel('Amostra', fontsize=10)
    ax2.set_ylabel('Valor Normalizado', fontsize=10)
    ax2.set_title('Predição vs Real (t+1)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Scatter plot
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(actuals[:, 0], predictions[:, 0], alpha=0.5, s=30)
    ax3.plot([actuals.min(), actuals.max()],
             [actuals.min(), actuals.max()], 'r--', linewidth=2)
    ax3.set_xlabel('Valor Real', fontsize=10)
    ax3.set_ylabel('Valor Predito', fontsize=10)
    ax3.set_title('Scatter Plot (t+1)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Calcular R²
    correlation = np.corrcoef(actuals[:, 0], predictions[:, 0])[0, 1]
    r2 = correlation ** 2
    ax3.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax3.transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 4. Exemplo de predição multi-step
    ax4 = fig.add_subplot(gs[1, 0])
    idx = min(50, len(actuals) - 1)
    steps = range(1, len(actuals[idx]) + 1)
    ax4.plot(steps, actuals[idx], 'o-', label='Real', markersize=8, linewidth=2)
    ax4.plot(steps, predictions[idx], 's-', label='Predito', markersize=8, linewidth=2)
    ax4.set_xlabel('Passos à Frente', fontsize=10)
    ax4.set_ylabel('Valor Normalizado', fontsize=10)
    ax4.set_title(f'Predição Multi-Step (Amostra {idx})', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Série original completa
    ax5 = fig.add_subplot(gs[1, 1:])
    dados_norm = dataset.scaler.transform(dados_originais)
    ax5.plot(dados_norm, label='Dados Históricos', linewidth=2, color='blue')
    ax5.axvline(x=len(dados_norm), color='red', linestyle='--', linewidth=2, label='Início da Predição')
    ax5.set_xlabel('Tempo', fontsize=10)
    ax5.set_ylabel('Preço Normalizado', fontsize=10)
    ax5.set_title('Série Temporal Completa', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Série futura gerada
    ax6 = fig.add_subplot(gs[2, :])
    # Últimos 100 pontos históricos
    ultimos_pontos = 100
    historico = dados_norm[-ultimos_pontos:]

    ax6.plot(range(len(historico)), historico, label='Histórico', linewidth=2, color='blue')
    ax6.plot(range(len(historico), len(historico) + len(serie_futura)),
             serie_futura, label='Predição Futura', linewidth=2, color='red', linestyle='--')
    ax6.axvline(x=len(historico), color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax6.fill_between(range(len(historico), len(historico) + len(serie_futura)),
                     serie_futura.flatten() - 0.5, serie_futura.flatten() + 0.5,
                     alpha=0.2, color='red', label='Intervalo de Incerteza')
    ax6.set_xlabel('Tempo', fontsize=10)
    ax6.set_ylabel('Preço Normalizado', fontsize=10)
    ax6.set_title('Projeção Futura (Geração Autoregressiva)', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle('TimeGPT - Análise Completa de Séries Temporais Financeiras',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.savefig('timegpt_analise_completa.png', dpi=300, bbox_inches='tight')
    print("\nGráficos salvos como 'timegpt_analise_completa.png'")
    plt.show()


# 8. Pipeline Principal
def main():
    # Hiperparâmetros otimizados para evitar overfitting
    TICKER = 'AAPL'
    SEQ_LEN = 60
    PRED_LEN = 5
    BATCH_SIZE = 64
    EPOCHS = 100

    # Carregar dados
    dados = carregar_dados(ticker=TICKER, periodo='2y')
    print(f"Dados carregados: {len(dados)} pontos")

    # Dividir em treino e validação
    train_size = int(0.8 * len(dados))
    train_data = dados[:train_size]
    val_data = dados[train_size:]

    # Criar datasets com augmentation no treino
    train_dataset = TimeSeriesDataset(train_data, seq_len=SEQ_LEN, pred_len=PRED_LEN, augment=True)
    val_dataset = TimeSeriesDataset(val_data, seq_len=SEQ_LEN, pred_len=PRED_LEN, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Amostras de treino: {len(train_dataset)}")
    print(f"Amostras de validação: {len(val_dataset)}")

    # Criar modelo com arquitetura reduzida para evitar overfitting
    model = TimeGPT(
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        pred_len=PRED_LEN
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModelo criado com {num_params:,} parâmetros")

    # Treinar
    train_losses, val_losses = treinar_modelo(model, train_loader, val_loader, epochs=EPOCHS)

    # Avaliar
    print("\nAvaliando modelo...")
    predictions, actuals = avaliar_modelo(model, val_dataset, num_predictions=100)

    # Calcular métricas
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8))) * 100

    print(f"\n{'=' * 50}")
    print(f"MÉTRICAS DE AVALIAÇÃO")
    print(f"{'=' * 50}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'=' * 50}")

    # Gerar série futura
    print("\nGerando série futura...")
    serie_futura = gerar_serie_futura(model, val_dataset, num_steps=50)
    print(f"Série futura gerada: {len(serie_futura)} passos")

    # Desnormalizar série futura para visualização
    serie_futura_denorm = val_dataset.scaler.inverse_transform(serie_futura.reshape(-1, 1))
    print(f"\nPrimeiros 5 valores preditos (desnormalizados):")
    for i, val in enumerate(serie_futura_denorm[:5], 1):
        print(f"  t+{i}: ${val[0]:.2f}")

    # Plotar resultados
    plotar_resultados_completos(predictions, actuals, serie_futura, dados,
                                train_losses, val_losses, val_dataset)

    # Salvar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': val_dataset.scaler,
        'seq_len': SEQ_LEN,
        'pred_len': PRED_LEN
    }, 'timegpt_model.pth')
    print("\nModelo salvo como 'timegpt_model.pth'")


if __name__ == "__main__":
    main()