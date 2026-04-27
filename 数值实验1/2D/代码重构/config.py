# config.py
import torch

# ----------------------------
# 设备与随机种子
# ----------------------------
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# 模型参数
# ----------------------------
HIDDEN_DIM = 512
NUM_HIDDEN_LAYERS = 5
ACTIVATION = 'tanh'
SIGMA = 10.0
FOURIER_DIM = 64

# ----------------------------
# 训练参数
# ----------------------------
LEARNING_RATE = 1e-4
NUM_EPOCHS_FULL = 50000
RESAMPLE_FREQ = 100
NUM_EPOCHS_SEARCH = 5000

# ----------------------------
# PDE 参数
# ----------------------------
EPS = 0.01
DOMAIN_SIZE = 1.0
T_FINAL = 10.0

# ----------------------------
# 超参数搜索
# ----------------------------
LAMBDA_CANDIDATES = [1.0, 10.0, 100.0]
LAMBDA_COMBINATIONS = [(bc, ic) for bc in LAMBDA_CANDIDATES for ic in [1.0, 10.0, 100.0]]