from easydict import EasyDict

ppo_forex_config = dict(
    exp_name='ppo_forex_trading',
    env=dict(
        env_id='ForexTradingEnv-v0',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=11000,
    ),
    policy=dict(
        cuda=False,
        on_policy=True,
        priority=False,
        model=dict(
            obs_shape=13,
            action_shape=3,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            update_per_collect=4,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            log_show_after_iter=10,
            log_show_metric=['value_loss', 'policy_entropy', 'explained_variance'],
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
        ),
        eval=dict(
            evaluator=dict(eval_freq=500),
        ),
    ),
    logger=dict(  #
        type='tensorboard',
        tensorboard_logger=dict(
            log_dir='./log/ppo_forex_trading',  # куда сохранять логи
            tag='ppo_forex',
        )
    ),
)
ppo_forex_config = EasyDict(ppo_forex_config)