import torch
import torch.nn as nn
from ppo_forex_config import ppo_forex_config
from ding.policy import PPOPolicy

# Определяем экспортируемую модель
class ExportableActor(nn.Module):
    def __init__(self, vac_model):
        super(ExportableActor, self).__init__()
        self.mlp = vac_model.actor[0]   # encoder
        self.head = vac_model.actor[1]  # head

    def forward(self, x):
        x = self.mlp(x)
        x = self.head(x)
        return x

def main():
    # Загрузка обученной политики
    ppo_forex_config.policy.multi_agent = False
    ppo_forex_config.policy.multi_gpu = False
    ppo_forex_config.policy.action_space = 'discrete'
    ppo_forex_config.policy.priority = False
    ppo_forex_config.policy.priority_IS_weight = 0.0
    ppo_forex_config.policy.collect.discount_factor = 0.99
    ppo_forex_config.policy.collect.gae_lambda = 0.95
    ppo_forex_config.policy.recompute_adv = False
    ppo_forex_config.policy.learn.ppo_param_init = True
    ppo_forex_config.policy.learn.grad_clip_type = 'clip_norm'
    ppo_forex_config.policy.learn.grad_clip_value = 0.5
    ppo_forex_config.policy.learn.lr_scheduler = None
    ppo_forex_config.policy.learn.value_norm = None
    policy = PPOPolicy(cfg=ppo_forex_config.policy)

    # Загрузка весов
    checkpoint_path = './exp/ppo_forex_trading/ckpt/ckpt_best.pth.tar'
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    policy._load_state_dict_eval(state_dict)

    # Обернуть модель для экспорта
    exportable_model = ExportableActor(policy._model)
    exportable_model.eval()  # Перевести в режим инференса

    # Создание входа
    obs_shape = ppo_forex_config.policy.model.obs_shape
    dummy_input = torch.randn(1, obs_shape)

    # Экспорт в ONNX
    onnx_path = 'ppo_forex_model.onnx'
    torch.onnx.export(
        exportable_model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11
    )

    print(f"Модель сохранена в {onnx_path}")

if __name__ == '__main__':
    main()