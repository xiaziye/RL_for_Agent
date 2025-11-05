import gymnasium as gym
import numpy as np
from utils import load_config

def main():
    config = load_config()
    model_path = config["model_save_path"]
    Q = np.load(model_path)

    env = gym.make(
        config["env"]["id"],
        is_slippery=config["env"]["is_slippery"],
        render_mode="human"
    )

    print("\n🎯 Final Evaluation (3 runs):")
    for i in range(3):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Run {i+1}: Reward = {total_reward}")
    env.close()

if __name__ == "__main__":
    main()