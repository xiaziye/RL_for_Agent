import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_config, ensure_dir

def main():
    config = load_config()
    model_path = config["model_save_path"]
    Q = np.load(model_path)

    env = gym.make(
        config["env"]["id"],
        is_slippery=config["env"]["is_slippery"],
        render_mode="rgb_array"
    )

    output_dir = "frozenlake_frames"
    ensure_dir(os.path.join(output_dir, "dummy"))  # 确保目录存在

    state, _ = env.reset()
    done = False
    step = 0
    action_names = ['L', 'D', 'R', 'U']

    while not done:
        action = int(np.argmax(Q[state]))
        frame = env.render()

        plt.figure(figsize=(4, 4))
        plt.imshow(frame)
        plt.axis('off')
        plt.title(f"Step {step} | Action: {action_names[action]}", fontsize=14)
        plt.savefig(
            os.path.join(output_dir, f"step_{step:03d}.png"),
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close()

        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1

    env.close()
    print(f"✅ Saved {step} frames to '{output_dir}/'")

if __name__ == "__main__":
    main()