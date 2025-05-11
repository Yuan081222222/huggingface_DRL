import gymnasium as gym # type: ignore

from stable_baselines3 import DQN # type: ignore
from stable_baselines3.common.env_util import make_vec_env # type: ignore
from stable_baselines3.common.evaluation import evaluate_policy # type: ignore
from stable_baselines3.common.monitor import Monitor # type: ignore

# Create environment
env = gym.make("LunarLander-v2")
env.reset()
env = make_vec_env('LunarLander-v2', n_envs=1)

# SOLUTION
# We added some parameters to accelerate the training
# model = DQN(
#     policy = 'MlpPolicy',
#     env = env,
#     learning_rate = 1e-3,
#     buffer_size = 100000,  # 緩衝區大小
#     learning_starts = 10000,  # 開始學習的步驟數
#     batch_size = 128, # 批次大小
#     gamma = 0.99, # discount factor
#     train_freq = 10,  # 每 4 個步驟訓練一次
#     gradient_steps = 1,  # 每次訓練更新 1 次
#     target_update_interval = 1000,  # 每 500 步更新一次目標網路
#     exploration_initial_eps =  0.9,  # 初始探索率
#     exploration_final_eps = 0.01,  # 最終探索率
#     verbose = 1
# )
model = DQN(
    policy='MlpPolicy',
    env=env,
    learning_rate=5e-4,  # 降低學習率以提高穩定性
    buffer_size=200000,  # 增加經驗回放緩衝區
    learning_starts=50000,  # 增加初始隨機收集的經驗
    batch_size=256,  # 增加批次大小
    gamma=0.99,  # 保持不變或嘗試 0.995
    train_freq=4,  # 每 4 步更新一次
    gradient_steps=1,
    target_update_interval=500,  # 降低目標網路更新間隔
    exploration_fraction=0.2,  # 控制探索率衰減速度
    exploration_initial_eps=1.0,  # 起始探索率調高
    exploration_final_eps=0.05,  # 終止探索率調高
    policy_kwargs=dict(net_arch=[256, 256]),  # 增大網路模型
    verbose=1
)

# SOLUTION
# Train it for 1,000,000 timesteps 1000000
model.learn(total_timesteps=2000000, progress_bar=True)
# Save the model
model_name = "dqn-LunarLander-v2"
model.save(model_name)

#@title
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")