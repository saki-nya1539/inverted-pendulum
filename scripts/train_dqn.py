"""
DQN (Deep Q-Network) エージェント訓練スクリプト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cart_pole_env import CartPoleEnv
from src.dqn_agent import DQNAgent
from src.train import train_dqn, save_training_results
from src.evaluate import Evaluator
from src.visualization import Visualizer
import numpy as np


def main():
    """
    DQN 訓練メイン処理
    """
    print("\n" + "=" * 70)
    print("🧠 DQN (Deep Q-Network) Agent Training")
    print("=" * 70)

    # ディレクトリ作成
    try:
        os.makedirs('data/models', exist_ok=True)
        os.makedirs('results/plots', exist_ok=True)
        print("✅ Directories created successfully")
    except Exception as e:
        print(f"❌ Error creating directories: {e}")
        return

    # 環境作成
    print("\n📌 Creating environment...")
    env = CartPoleEnv()

    # エージェント作成
    print("📌 Creating DQN agent...")
    agent = DQNAgent(
        state_size=4,
        action_size=2,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        memory_size=2000
    )

    # 訓練
    print("\n📌 Starting training...")
    rewards, episode_lengths, losses = train_dqn(
        env=env,
        agent=agent,
        num_episodes=500,
        render=False
    )

    # 結果を保存
    print("\n📌 Saving results...")
    save_training_results(rewards, episode_lengths, agent, save_dir='data/models')

    # 評価
    print("\n📌 Evaluating agent...")
    evaluator = Evaluator(agent, env)
    stats = evaluator.evaluate(num_episodes=20)
    print(f"\n📊 Evaluation Results:")
    print(f"   Mean Reward: {stats['mean_reward']:.2f}")
    print(f"   Max Reward: {stats['max_reward']:.2f}")
    print(f"   Min Reward: {stats['min_reward']:.2f}")
    print(f"   Std Dev: {stats['std_reward']:.2f}")

    # 可視化
    print("\n📌 Creating visualizations...")
    viz = Visualizer(save_path='results/plots')
    viz.plot_training_history(
        rewards,
        title="DQN Training History",
        filename="dqn_training.png"
    )
    viz.plot_episode_length_distribution(
        episode_lengths,
        title="DQN Episode Length Distribution",
        filename="dqn_episode_lengths.png"
    )

    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"📁 Models saved to: data/models/")
    print(f"📁 Plots saved to: results/plots/")
    print("=" * 70 + "\n")

    env.close()


if __name__ == "__main__":
    main()