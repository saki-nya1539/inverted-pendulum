"""
Q-Learning エージェント訓練スクリプト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cart_pole_env import CartPoleEnv
from src.q_learning_agent import QLearningAgent
from src.train import train_q_learning, save_training_results
from src.evaluate import Evaluator
from src.visualization import Visualizer
import numpy as np


def main():
    """
    Q-Learning 訓練メイン処理
    """
    print("\n" + "=" * 70)
    print("🤖 Q-Learning Agent Training")
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
    print("📌 Creating Q-Learning agent...")
    agent = QLearningAgent(
        num_states=20,
        num_actions=2,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )

    # 訓練
    print("\n📌 Starting training...")
    rewards, episode_lengths = train_q_learning(
        env=env,
        agent=agent,
        num_episodes=1000,
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
        title="Q-Learning Training History",
        filename="q_learning_training.png"
    )
    viz.plot_episode_length_distribution(
        episode_lengths,
        title="Q-Learning Episode Length Distribution",
        filename="q_learning_episode_lengths.png"
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