import torch

# --------------------------
# Configs / Reprodutibilidade
# --------------------------
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------
# Parâmetros
# --------------------------
TICKER = "PETR4.SA"       # alterar se quiser outro ativo
START = "2018-01-01"
END = "2023-12-31"
SEQ_LEN = 50
BATCH_SIZE = 64
EPOCHS = 12               # aumentar melhora o ajuste
LR = 1e-3
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
FUTURE_STEPS = 200        # quantos pontos gerar
N_GENERATED_SERIES = 5    # quantas trajetórias gerar
MIN_SIGMA = 1e-4          # estabilidade numérica
