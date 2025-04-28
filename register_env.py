import gym
from gym.envs.registration import register

register(
    id='ForexTradingEnv-v0',  # должно совпадать с env_id в конфиге
    entry_point='environment:ForexTradingEnv',  # environment.py — имя файла, ForexTradingEnv — имя класса
)