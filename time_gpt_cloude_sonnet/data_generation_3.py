import torch
import torch.nn as nn
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math
import warnings

warnings.filterwarnings('ignore')

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {device}")


# Definir arquitetura do modelo (deve ser igual ao código de treino)
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

        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

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

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x[:, -10:, :], dim=1)
        output = self.output_proj(x)
        output = output.reshape(-1, self.pred_len, 1)
        return output


# Funções de geração de séries
def carregar_modelo(caminho='timegpt_model.pth'):
    """Carrega o modelo treinado e o scaler"""
    print(f"Carregando modelo de: {caminho}")

    # Adicionar StandardScaler como safe global para PyTorch 2.6+
    try:
        from sklearn.preprocessing._data import StandardScaler as SKStandardScaler
        torch.serialization.add_safe_globals([SKStandardScaler])
    except:
        pass

    # Carregar com weights_only=False (seguro se você confia no arquivo)
    checkpoint = torch.load(caminho, map_location=device, weights_only=False)

    # Criar modelo com mesma arquitetura
    model = TimeGPT(
        input_dim=1,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2,
        pred_len=checkpoint['pred_len']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Modelo carregado com sucesso!")
    print(f"  SEQ_LEN: {checkpoint['seq_len']}, PRED_LEN: {checkpoint['pred_len']}")

    return model, checkpoint['scaler'], checkpoint['seq_len'], checkpoint['pred_len']


def carregar_dados_base(ticker='AAPL', periodo='2y'):
    """Carrega dados históricos para usar como seed"""
    print(f"\nBaixando dados base de {ticker}...")
    dados = yf.download(ticker, period=periodo, interval='1d', progress=False)
    return dados['Close'].values.reshape(-1, 1)


def gerar_serie_com_variacao(model, sequencia_inicial, scaler, seq_len,
                             num_steps=100, temperatura=1.0, adicionar_ruido=True,
                             momentum=0.0, volatilidade_crescente=False,
                             injecao_shock=False):
    """
    Gera uma série temporal com variações estocásticas e não-estacionariedade

    Args:
        model: modelo treinado
        sequencia_inicial: sequência histórica inicial
        scaler: StandardScaler usado no treino
        seq_len: tamanho da janela de entrada
        num_steps: quantos passos gerar
        temperatura: controla a aleatoriedade (maior = mais variação)
        adicionar_ruido: se True, adiciona ruído gaussiano às predições
        momentum: adiciona tendência persistente (-1 a 1)
        volatilidade_crescente: se True, aumenta volatilidade ao longo do tempo
        injecao_shock: se True, adiciona shocks aleatórios periódicos
    """
    model.eval()

    # Normalizar sequência inicial
    sequencia = scaler.transform(sequencia_inicial.copy())
    predicoes = []

    # Variáveis para não-estacionariedade
    tendencia_acumulada = 0
    volatilidade_base = 0.05
    ultimo_retorno = 0

    with torch.no_grad():
        for step in range(num_steps):
            # Pegar últimos seq_len pontos
            x_input = torch.FloatTensor(sequencia[-seq_len:]).unsqueeze(0).to(device)

            # Predizer
            pred = model(x_input)
            proximo_valor = pred[0, 0].cpu().numpy()

            # Aplicar temperatura (aumenta variação)
            if temperatura != 1.0:
                proximo_valor = proximo_valor * temperatura

            # ==== MECANISMOS PARA EVITAR CONVERGÊNCIA ====

            # 1. Momentum - adiciona persistência de tendência
            if momentum != 0:
                tendencia = np.sign(proximo_valor - sequencia[-1]) * abs(momentum) * 0.1
                tendencia_acumulada = 0.7 * tendencia_acumulada + 0.3 * tendencia
                proximo_valor = proximo_valor + tendencia_acumulada

            # 2. Volatilidade crescente ou variável
            if volatilidade_crescente:
                # Volatilidade aumenta com o tempo
                vol_atual = volatilidade_base * (1 + step / num_steps)
            else:
                # Volatilidade com regime switching (simula mudanças de mercado)
                if step % 20 == 0:  # Muda regime a cada 20 passos
                    volatilidade_base = np.random.uniform(0.03, 0.12)
                vol_atual = volatilidade_base

            # 3. Ruído heterocedástico (volatilidade variável)
            if adicionar_ruido:
                # GARCH-like: volatilidade depende de retornos passados
                retorno_atual = abs(proximo_valor - sequencia[-1])
                vol_garch = 0.7 * vol_atual + 0.3 * abs(ultimo_retorno) * temperatura
                ruido = np.random.normal(0, vol_garch, proximo_valor.shape)
                proximo_valor = proximo_valor + ruido
                ultimo_retorno = retorno_atual

            # 4. Injeção de shocks aleatórios (eventos extremos)
            if injecao_shock:
                # 5% de chance de shock em cada passo
                if np.random.rand() < 0.05:
                    magnitude_shock = np.random.choice([-1, 1]) * np.random.uniform(0.1, 0.3)
                    proximo_valor = proximo_valor + magnitude_shock
                    print(f"  💥 Shock aplicado no passo {step}: {magnitude_shock:.3f}")

            # 5. Drift aleatório (tendência de longo prazo)
            drift = np.random.normal(0, 0.01 * temperatura)
            proximo_valor = proximo_valor + drift

            # 6. Mean reversion fraco (evita explosão mas mantém volatilidade)
            media_recente = np.mean(sequencia[-20:])
            forca_reversion = 0.02  # Fraco para não estacionar
            reversion = (media_recente - proximo_valor) * forca_reversion
            proximo_valor = proximo_valor + reversion

            # Adicionar à sequência
            predicoes.append(proximo_valor)
            sequencia = np.vstack([sequencia, proximo_valor.reshape(1, -1)])

    # Desnormalizar
    predicoes_array = np.array(predicoes).reshape(-1, 1)
    predicoes_denorm = scaler.inverse_transform(predicoes_array)

    return predicoes_denorm


def gerar_multiplas_series(model, dados_base, scaler, seq_len,
                           num_series=10, num_steps=100,
                           temperaturas=None, usar_diferentes_seeds=True,
                           com_diversidade=True):
    """
    Gera múltiplas séries temporais com diferentes configurações e não-estacionariedade

    Args:
        num_series: número de séries a gerar
        temperaturas: lista de temperaturas (se None, usa variação aleatória)
        usar_diferentes_seeds: se True, usa diferentes pontos iniciais dos dados
        com_diversidade: se True, adiciona mecanismos para evitar convergência
    """
    print(f"\n{'=' * 60}")
    print(f"GERANDO {num_series} SÉRIES TEMPORAIS COM TENDÊNCIAS VARIADAS")
    print(f"{'=' * 60}")

    series_geradas = []
    configs = []

    # Se não especificado, gera temperaturas aleatórias mais altas
    if temperaturas is None:
        temperaturas = np.random.uniform(0.8, 1.5, num_series)

    # Tipos de tendências disponíveis
    tipos_tendencia = [
        'alta',  # Tendência linear de alta
        'baixa',  # Tendência linear de baixa
        'exponencial_alta',  # Bull market exponencial
        'exponencial_baixa',  # Bear market / crash
        'senoidal',  # Ciclos bull/bear
        'parabolica_alta',  # Bubble formation
        'parabolica_baixa',  # Bubble burst
        'logaritmica_alta',  # Crescimento maduro
        'em_v',  # V-shape recovery
        'em_v_invertido',  # Inverted V (peak)
        'escada_alta',  # Breakouts em níveis
        'escada_baixa',  # Breakdowns em níveis
        'neutro'  # Random walk
    ]

    for i in range(num_series):
        print(f"\n🔄 Gerando série {i + 1}/{num_series}...")

        # Escolher ponto inicial
        if usar_diferentes_seeds and len(dados_base) > seq_len * 2:
            max_start = len(dados_base) - seq_len - 1
            start_idx = np.random.randint(max_start // 2, max_start)
            sequencia_inicial = dados_base[start_idx:start_idx + seq_len]
        else:
            sequencia_inicial = dados_base[-seq_len:]

        # Configuração diversificada para cada série
        temp = temperaturas[i] if i < len(temperaturas) else 1.2

        if com_diversidade:
            # Selecionar tipo de tendência (ciclicamente ou aleatório)
            tipo_tend = tipos_tendencia[i % len(tipos_tendencia)]

            # Força da tendência variável
            forca = np.random.uniform(0.3, 1.0)

            # Configurações baseadas no tipo de tendência
            if 'exponencial' in tipo_tend or 'parabolica' in tipo_tend:
                # Tendências fortes precisam menos momentum adicional
                momentum_val = np.random.uniform(-0.1, 0.1)
                shock = True
                vol_cresc = np.random.rand() > 0.5
            elif 'senoidal' in tipo_tend:
                # Ciclos não precisam momentum
                momentum_val = 0
                shock = True
                vol_cresc = False
            elif 'escada' in tipo_tend:
                # Escadas com pouco momentum
                momentum_val = np.random.uniform(-0.2, 0.2)
                shock = True
                vol_cresc = False
            elif tipo_tend in ['alta', 'baixa']:
                # Tendências lineares podem ter momentum
                momentum_val = 0.3 if tipo_tend == 'alta' else -0.3
                shock = True
                vol_cresc = np.random.rand() > 0.7
            else:  # neutro
                momentum_val = 0
                shock = True
                vol_cresc = False

            comportamento = {
                'momentum': momentum_val,
                'vol_crescente': vol_cresc,
                'shock': shock,
                'ruido': True,
                'tipo_tendencia': tipo_tend,
                'forca_tendencia': forca
            }
        else:
            # Comportamento padrão
            comportamento = {
                'momentum': 0,
                'vol_crescente': False,
                'shock': False,
                'ruido': True,
                'tipo_tendencia': 'neutro',
                'forca_tendencia': 0.5
            }

        config = {
            'temperatura': temp,
            'seed_idx': start_idx if usar_diferentes_seeds else -seq_len,
            **comportamento
        }
        configs.append(config)

        # Gerar série com nova função
        serie = gerar_serie_com_variacao(
            model, sequencia_inicial, scaler, seq_len,
            num_steps=num_steps,
            temperatura=temp,
            adicionar_ruido=comportamento['ruido'],
            momentum=comportamento['momentum'],
            volatilidade_crescente=comportamento['vol_crescente'],
            injecao_shock=comportamento['shock'],
            tipo_tendencia=comportamento['tipo_tendencia'],
            forca_tendencia=comportamento['forca_tendencia']
        )

        series_geradas.append(serie)

        # Estatísticas da série
        retornos = np.diff(serie.flatten()) / serie[:-1].flatten()

        # Emoji baseado no tipo de tendência
        emoji_map = {
            'alta': '📈', 'baixa': '📉', 'exponencial_alta': '🚀',
            'exponencial_baixa': '💥', 'senoidal': '〰️', 'parabolica_alta': '🎢',
            'parabolica_baixa': '⬇️', 'logaritmica_alta': '📊', 'em_v': '✅',
            'em_v_invertido': '⚠️', 'escada_alta': '🪜⬆️', 'escada_baixa': '🪜⬇️',
            'neutro': '➡️'
        }
        emoji = emoji_map.get(comportamento['tipo_tendencia'], '📊')

        print(f"  {emoji} Tipo: {comportamento['tipo_tendencia'].upper()}")
        print(f"  ✓ Força: {comportamento['forca_tendencia']:.2f}")
        print(f"  ✓ Temperatura: {temp:.2f}")
        print(f"  ✓ Momentum: {comportamento['momentum']:+.2f}")
        print(f"  ✓ Vol. Crescente: {'Sim' if comportamento['vol_crescente'] else 'Não'}")
        print(f"  ✓ Shocks: {'Sim' if comportamento['shock'] else 'Não'}")
        print(f"  ✓ Valor inicial: ${serie[0][0]:.2f}, Valor final: ${serie[-1][0]:.2f}")
        print(f"  ✓ Variação: {((serie[-1][0] - serie[0][0]) / serie[0][0] * 100):+.2f}%")
        print(f"  ✓ Volatilidade média: {np.std(retornos) * 100:.2f}%")

    return series_geradas, configs


def plotar_series_multiplas(series_geradas, dados_base, configs, scaler):
    """Cria visualizações completas das séries geradas"""

    num_series = len(series_geradas)

    # Figura 1: Todas as séries juntas
    fig1, ax1 = plt.subplots(figsize=(16, 8))

    # Plotar histórico
    historico_denorm = scaler.inverse_transform(scaler.transform(dados_base))
    ultimos_100 = historico_denorm[-100:]
    ax1.plot(range(len(ultimos_100)), ultimos_100,
             linewidth=3, color='black', label='Histórico Real', alpha=0.8)

    # Plotar cada série gerada
    cores = plt.cm.rainbow(np.linspace(0, 1, num_series))
    for i, (serie, cor) in enumerate(zip(series_geradas, cores)):
        x_range = range(len(ultimos_100), len(ultimos_100) + len(serie))
        temp = configs[i]['temperatura']
        ax1.plot(x_range, serie, linewidth=2, alpha=0.7, color=cor,
                 label=f'Série {i + 1} (T={temp:.2f})')

    ax1.axvline(x=len(ultimos_100), color='red', linestyle='--',
                linewidth=2, label='Início da Geração')
    ax1.set_xlabel('Tempo (dias)', fontsize=12)
    ax1.set_ylabel('Preço ($)', fontsize=12)
    ax1.set_title('Múltiplas Séries Temporais Geradas - Visão Geral',
                  fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('series_multiplas_geral.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico geral salvo: 'series_multiplas_geral.png'")
    plt.show()

    # Figura 2: Grid com cada série separada
    n_cols = 3
    n_rows = (num_series + n_cols - 1) // n_cols
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten() if num_series > 1 else [axes]

    for i, (serie, ax, config) in enumerate(zip(series_geradas, axes, configs)):
        # Contexto histórico
        contexto = 50
        hist_contexto = historico_denorm[-contexto:]
        ax.plot(range(len(hist_contexto)), hist_contexto,
                linewidth=2, color='blue', alpha=0.6, label='Histórico')

        # Série gerada
        x_range = range(len(hist_contexto), len(hist_contexto) + len(serie))
        ax.plot(x_range, serie, linewidth=2.5, color='red',
                alpha=0.8, label='Gerado')

        # Linha divisória
        ax.axvline(x=len(hist_contexto), color='green',
                   linestyle='--', linewidth=2, alpha=0.7)

        # Estatísticas
        var_pct = ((serie[-1][0] - serie[0][0]) / serie[0][0] * 100)
        media = np.mean(serie)
        std = np.std(serie)

        # Título com info
        ax.set_title(f'Série {i + 1} | T={config["temperatura"]:.2f} | '
                     f'Var: {var_pct:+.1f}%',
                     fontsize=11, fontweight='bold')

        # Texto com estatísticas
        stats_text = f'Média: ${media:.2f}\nStd: ${std:.2f}\n'
        stats_text += f'Min: ${np.min(serie):.2f}\nMax: ${np.max(serie):.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        ax.set_xlabel('Tempo', fontsize=10)
        ax.set_ylabel('Preço ($)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Esconder eixos extras
    for i in range(num_series, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Séries Temporais Geradas Individualmente',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('series_multiplas_individual.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico individual salvo: 'series_multiplas_individual.png'")
    plt.show()

    # Figura 3: Análise estatística
    fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))

    # Distribuição de retornos
    ax_ret = axes3[0, 0]
    for i, serie in enumerate(series_geradas):
        retornos = np.diff(serie.flatten()) / serie[:-1].flatten() * 100
        ax_ret.hist(retornos, bins=30, alpha=0.5, label=f'Série {i + 1}')
    ax_ret.set_xlabel('Retorno (%)', fontsize=10)
    ax_ret.set_ylabel('Frequência', fontsize=10)
    ax_ret.set_title('Distribuição de Retornos', fontsize=12, fontweight='bold')
    ax_ret.legend(fontsize=8)
    ax_ret.grid(True, alpha=0.3)

    # Volatilidade ao longo do tempo
    ax_vol = axes3[0, 1]
    for i, serie in enumerate(series_geradas):
        retornos = np.diff(serie.flatten()) / serie[:-1].flatten()
        vol_rolling = np.array([np.std(retornos[max(0, j - 10):j + 1])
                                for j in range(len(retornos))])
        ax_vol.plot(vol_rolling, label=f'Série {i + 1}', alpha=0.7)
    ax_vol.set_xlabel('Tempo', fontsize=10)
    ax_vol.set_ylabel('Volatilidade', fontsize=10)
    ax_vol.set_title('Volatilidade Rolling (janela=10)', fontsize=12, fontweight='bold')
    ax_vol.legend(fontsize=8)
    ax_vol.grid(True, alpha=0.3)

    # Comparação de trajetórias normalizadas
    ax_norm = axes3[1, 0]
    for i, serie in enumerate(series_geradas):
        serie_norm = (serie - serie[0]) / serie[0] * 100
        ax_norm.plot(serie_norm, label=f'Série {i + 1}', alpha=0.7)
    ax_norm.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax_norm.set_xlabel('Tempo', fontsize=10)
    ax_norm.set_ylabel('Variação Percentual (%)', fontsize=10)
    ax_norm.set_title('Trajetórias Normalizadas (base=0)', fontsize=12, fontweight='bold')
    ax_norm.legend(fontsize=8)
    ax_norm.grid(True, alpha=0.3)

    # Box plot comparativo
    ax_box = axes3[1, 1]
    dados_box = [serie.flatten() for serie in series_geradas]
    bp = ax_box.boxplot(dados_box, labels=[f'S{i + 1}' for i in range(num_series)],
                        patch_artist=True)
    for patch, cor in zip(bp['boxes'], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.6)
    ax_box.set_xlabel('Série', fontsize=10)
    ax_box.set_ylabel('Preço ($)', fontsize=10)
    ax_box.set_title('Distribuição de Preços por Série', fontsize=12, fontweight='bold')
    ax_box.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('series_multiplas_analise.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico de análise salvo: 'series_multiplas_analise.png'")
    plt.show()


def exibir_estatisticas_gerais(series_geradas, configs):
    """Exibe estatísticas comparativas de todas as séries"""
    print(f"\n{'=' * 70}")
    print(f"ESTATÍSTICAS COMPARATIVAS DAS SÉRIES NÃO-ESTACIONÁRIAS")
    print(f"{'=' * 70}")

    for i, (serie, config) in enumerate(zip(series_geradas, configs), 1):
        var_pct = ((serie[-1][0] - serie[0][0]) / serie[0][0] * 100)
        retornos = np.diff(serie.flatten()) / serie[:-1].flatten()

        # Calcular Max Drawdown
        serie_flat = serie.flatten()
        cummax = np.maximum.accumulate(serie_flat)
        drawdown = (serie_flat - cummax) / cummax
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # Calcular Sharpe Ratio
        sharpe = np.mean(retornos) / np.std(retornos) if np.std(retornos) > 0 else 0

        # Teste de estacionariedade simples (variância em janelas)
        n_windows = 5
        window_size = len(serie) // n_windows
        variancias = []
        for j in range(n_windows):
            start = j * window_size
            end = start + window_size
            if end <= len(serie):
                window_data = serie[start:end].flatten()
                variancias.append(np.var(window_data))

        razao_var = max(variancias) / min(variancias) if min(variancias) > 0 else 1
        estacionaria = "Sim" if razao_var < 2 else "Não"

        print(f"\n📊 Série {i}:")
        print(f"   Config: Temp={config['temperatura']:.2f}, Mom={config.get('momentum', 0):+.2f}, "
              f"VolCresc={'S' if config.get('vol_crescente', False) else 'N'}, "
              f"Shock={'S' if config.get('shock', False) else 'N'}")
        print(f"   Preço inicial: ${serie[0][0]:.2f}")
        print(f"   Preço final: ${serie[-1][0]:.2f}")
        print(f"   Variação total: {var_pct:+.2f}%")
        print(f"   Retorno médio: {np.mean(retornos) * 100:.3f}%")
        print(f"   Volatilidade: {np.std(retornos) * 100:.3f}%")
        print(f"   Sharpe Ratio: {sharpe:.3f}")
        print(f"   Max Drawdown: {max_drawdown:.2%}")
        print(f"   Razão Variância (janelas): {razao_var:.2f}")
        print(f"   Aparentemente estacionária: {estacionaria}")


def main():
    # Configurações
    MODELO_PATH = 'timegpt_model.pth'
    TICKER = 'AAPL'
    NUM_SERIES = 9  # Número de séries a gerar
    NUM_STEPS = 100  # Passos futuros para cada série

    print("=" * 70)
    print("GERADOR DE MÚLTIPLAS SÉRIES TEMPORAIS - TimeGPT")
    print("=" * 70)

    # Carregar modelo
    model, scaler, seq_len, pred_len = carregar_modelo(MODELO_PATH)

    # Carregar dados base
    dados_base = carregar_dados_base(ticker=TICKER)

    # Gerar múltiplas séries com diferentes configurações
    series_geradas, configs = gerar_multiplas_series(
        model, dados_base, scaler, seq_len,
        num_series=NUM_SERIES,
        num_steps=NUM_STEPS,
        temperaturas=None,  # None = aleatório
        usar_diferentes_seeds=True,
        com_diversidade=True  # TRUE = evita convergência
    )

    # Exibir estatísticas
    exibir_estatisticas_gerais(series_geradas, configs)

    # Plotar todas as visualizações
    print(f"\n{'=' * 70}")
    print("GERANDO VISUALIZAÇÕES")
    print(f"{'=' * 70}")
    plotar_series_multiplas(series_geradas, dados_base, configs, scaler)

    # Salvar séries em arquivo
    print(f"\n{'=' * 70}")
    print("SALVANDO DADOS")
    print(f"{'=' * 70}")
    np.savez('series_geradas.npz',
             series=[s for s in series_geradas],
             configs=configs,
             dados_base=dados_base)
    print("✓ Séries salvas em 'series_geradas.npz'")

    print(f"\n{'=' * 70}")
    print("✓ PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"{'=' * 70}")
    print(f"\nArquivos gerados:")
    print(f"  1. series_multiplas_geral.png")
    print(f"  2. series_multiplas_individual.png")
    print(f"  3. series_multiplas_analise.png")
    print(f"  4. series_geradas.npz")


if __name__ == "__main__":
    main()