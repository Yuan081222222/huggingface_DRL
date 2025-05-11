import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from huggingface_sb3 import package_to_hub

## Define a repo_id
repo_id = "Yuan081222222/dqn-LunarLander-v2"  # 請替換成您的 Hugging Face 用戶名

# Define the name of the environment
env_id = "LunarLander-v2"

# 定義模型名稱和路徑
model_name = "dqn-LunarLander-v2"
model_path = "dqn-LunarLander-v2.zip"  # 假設檔案在同一目錄下

# 載入已訓練的模型
model = DQN.load(model_path)

# Create the evaluation env and set the render_mode="rgb_array"
eval_env = DummyVecEnv([lambda: Monitor(gym.make(env_id, render_mode="rgb_array"))])

# Define the model architecture we used
model_architecture = "DQN"

## Define the commit message
commit_message = "unit1"

# Push to hub
package_to_hub(model=model,  # 使用載入的模型對象
               model_name=model_name,
               model_architecture=model_architecture,
               env_id=env_id,
               eval_env=eval_env,
               repo_id=repo_id,
               commit_message=commit_message)
