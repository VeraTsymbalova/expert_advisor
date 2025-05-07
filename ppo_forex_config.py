from easydict import EasyDict

ppo_forex_config = dict(
    exp_name='ppo_forex_trading',
    env=dict(
        env_id='ForexTradingEnv-v0',
        collector_env_num=8, # Количество сред для сбора данных (параллельно)
        evaluator_env_num=8, # Количество сред для оценки политики
        n_evaluator_episode=8, # Количество эпизодов при каждой итерации оценки
        stop_value=14000, # Значение net worth, при достижении которого обучение завершится
    ),
    policy=dict(
        cuda=False, # Использовать ли GPU (True/False)
        on_policy=True, # Используется on-policy алгоритм (PPO относится к таким)
        priority=False, # Приоритетная выборка не используется
        model=dict(
            obs_shape=13, # Размерность вектора наблюдения (входные признаки среды)
            action_shape=3, # Количество возможных действий (0-hold, 1-buy, 2-sell)
            encoder_hidden_size_list=[128, 128, 64], # Архитектура скрытых слоев нейросети
        ),
        learn=dict(
            update_per_collect=4, # Сколько раз обновлять модель после каждого сбора опыта
            batch_size=64, # Размер мини-батча
            learning_rate=3e-4, # Скорость обучения
            value_weight=0.5, # Вес функции потерь критика (оценка ценности)
            entropy_weight=0.05, # Вес энтропийного бонуса (поощрение разнообразия действий)
            clip_ratio=0.2, # Порог обрезки вероятностей (важно для стабильности PPO)
            adv_norm=True, # Нормализация оценок преимущества
            log_show_after_iter=10, # Периодичность отображения логов
            epoch_per_collect=8,         # Количество эпох обучения после каждой итерации сбора опыта
            max_sample=50000,            #  Ограничение на количество обучающих примеров
            max_iteration=1000,          #  Максимальное количество итераций обучения
            log_show_metric=[            # Метрики, которые будут логироваться
                'value_loss',
                'policy_entropy',
                'reward_mean',
                'return_mean',
                'grad_norm',
                'adv_max',
                'episode_length'
            ],
        ),
        collect=dict(
            n_sample=128, # Количество собранных шагов перед обновлением модели
            unroll_len=1, # Длина разворачивания последовательности действий (обычно = 1 для PPO)
        ),
        eval=dict(
            evaluator=dict(eval_freq=100), # Частота оценки модели (каждые N итераций)
        ),
    ),
    logger=dict(  #
        type='tensorboard',  # Используемый инструмент логирования
        tensorboard_logger=dict(
            log_dir='./log/ppo_forex_trading',
            tag='ppo_forex',
        )
    ),
)
ppo_forex_config = EasyDict(ppo_forex_config)
