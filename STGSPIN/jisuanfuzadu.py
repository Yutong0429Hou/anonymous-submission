import torch
import torch.profiler
from model import create_model
from options.train_options import TrainOptions
# 创建模型
opt = TrainOptions().parse()
model = create_model(opt)
model.eval()  # 评估模式

# 计算总参数量
total_params = 0
for net_name in model.model_names:  # 遍历 PR 中的所有子网络
    net = getattr(model, f"net_{net_name}", None)  # 获取对应的子网络
    if net and isinstance(net, torch.nn.Module):  # 确保是 PyTorch 模型
        net_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        total_params += net_params
        print(f"Model {net_name} Parameters: {net_params / 1e6:.2f}M")

print(f"Total Parameters in PR Model: {total_params / 1e6:.2f}M")


