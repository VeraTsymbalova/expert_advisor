from easydict import EasyDict

ppo_forex_create_config = dict(
    env=dict(
        type='gym',  # 
        env_id='ForexTradingEnv-v0',
        import_names=['environment'],
    ),
    env_manager=dict(
        type='base',  # 
    ),
    policy=dict(
        type='ppo',
    ),
)
ppo_forex_create_config = EasyDict(ppo_forex_create_config)