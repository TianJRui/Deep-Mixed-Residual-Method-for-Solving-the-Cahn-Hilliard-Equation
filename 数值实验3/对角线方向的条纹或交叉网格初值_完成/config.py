# config_heat.py
import torch

# ----------------------------
# 设备与随机种子
# ----------------------------
SEED = 65335454
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------
# 模型参数
# ----------------------------
HIDDEN_DIM = 512
NUM_HIDDEN_LAYERS = 4
ACTIVATION = 'tanh'

# ----------------------------
# 训练参数
# ----------------------------
LEARNING_RATE = 1e-2
LEARNING_RATE_FINE_TUNING = 1e-2
RESAMPLE_FREQ = 5
NUM_EPOCHS = 30000
T_FINAL = 0.5
T_REVERSE = 0.0
# ----------------------------
# PDE 参数
# ----------------------------
LAMBDA_BC = 1.0     # 边界损失权重
LAMBDA_IC = 1.0    # 初始条件损失权重
DOMAIN_SIZE = 2.0    # 域大小 [-1, 1]^2
# ----------------------------
# 正则化参数
# ----------------------------
EPS_REG_INITIAL = 1e-2      # 初始正则化强度
EPS_REG_MIN = 1e-6         # 最小正则化值
EPS_REG_DECAY_RATE = 5e-4  # 衰减速率
gauss_n = 20