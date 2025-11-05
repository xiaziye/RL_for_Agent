import gymnasium as gym
import numpy as np
import logging
from utils import load_config, ensure_dir

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    config = load_config()
    env_config = config["env"]
    train_config = config["training"]
    model_path = config["model_save_path"]

    env = gym.make(
        env_config["id"],
        is_slippery=env_config["is_slippery"],
        render_mode=env_config["render_mode"]
    )

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    alpha = train_config["alpha"]
    gamma = train_config["gamma"]
    epsilon = train_config["epsilon_start"]
    eps_decay = train_config["epsilon_decay"]
    min_eps = train_config["epsilon_min"]
    episodes = train_config["episodes"]
    eval_interval = train_config["eval_interval"]
    target_avg = train_config["target_avg_reward"]

    scores = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        while True:
            if np.random.uniform() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )

            total_reward += reward
            state = next_state

            if done:
                break

        scores.append(total_reward)
        epsilon = max(min_eps, eps_decay * epsilon)

        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(scores[-eval_interval:])
            logger.info(f"Episode {episode + 1:4d} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.2f}")
            if avg_reward >= target_avg:
                logger.info("✅ Q-Table converged!")
                break

    ensure_dir(model_path)
    np.save(model_path, Q)
    env.close()
    logger.info(f"✅ Q-table saved to {model_path}")

if __name__ == "__main__":
    main()