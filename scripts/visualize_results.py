"""
訓練結果を可視化するスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import Visualizer
import pickle
import numpy as np


def load_history(model_name: str, history_path: str = 'data/models'):
    """
    訓練履歴を読み込む
    
    Parameters
    ----------
    model_name : str
        モデル名 ("qlearningagent" または "dqnagent")
    history_path : str
        履歴ファイルパス
        
    Returns
    -------
    dict or None
        履歴データまたは None
    """
    path = os.path.join(history_path, f'{model_name}_history.pkl')
    
    if not os.path.exists(path):
        print(f"❌ History not found: {path}")
        return None
    
    with open(path, 'rb') as f:
        history = pickle.load(f)
    print(f"✅ Loaded history from {path}")
    return history


def main():
    """
    可視化メイン処理
    """
    print("\n" + "="*70)
    print("📊 Visualizing Training Results")
    print("="*70)
    
    # ディレクトリ作成
    os.makedirs('results/plots', exist_ok=True)
    
    # Visualizer を作成
    viz = Visualizer(save_path='results/plots')
    
    # 訓練履歴を読み込み
    print("\n📌 Loading training histories...")
    
    q_learning_history = load_history('qlearningagent')
    dqn_history = load_history('dqnagent')
    
    if not q_learning_history and not dqn_history:
        print("\n❌ No training histories found!")
        print("Please run training scripts first:")
        print("  python scripts/train_q_learning.py")
        print("  python scripts/train_dqn.py")
        return
    
    # Q-Learning の可視化
    if q_learning_history:
        print("\n📌 Creating Q-Learning visualizations...")
        
        rewards = q_learning_history['rewards']
        lengths = q_learning_history['episode_lengths']
        
        viz.plot_training_history(
            rewards,
            title="Q-Learning Training History",
            filename="q_learning_training_detailed.png"
        )
        
        viz.plot_episode_length_distribution(
            lengths,
            title="Q-Learning Episode Length Distribution",
            filename="q_learning_episode_dist.png"
        )
        
        print(f"   Final Mean Reward: {np.mean(rewards[-100:]):.2f}")
        print(f"   Max Reward: {np.max(rewards):.2f}")
    
    # DQN の可視化
    if dqn_history:
        print("\n📌 Creating DQN visualizations...")
        
        rewards = dqn_history['rewards']
        lengths = dqn_history['episode_lengths']
        
        viz.plot_training_history(
            rewards,
            title="DQN Training History",
            filename="dqn_training_detailed.png"
        )
        
        viz.plot_episode_length_distribution(
            lengths,
            title="DQN Episode Length Distribution",
            filename="dqn_episode_dist.png"
        )
        
        print(f"   Final Mean Reward: {np.mean(rewards[-100:]):.2f}")
        print(f"   Max Reward: {np.max(rewards):.2f}")
    
    # 比較グラフ
    if q_learning_history and dqn_history:
        print("\n📌 Creating comparison plots...")
        
        comparison_data = {
            'Q-Learning': q_learning_history['rewards'],
            'DQN': dqn_history['rewards']
        }
        
        viz.plot_comparison(
            comparison_data,
            title="Q-Learning vs DQN Comparison",
            filename="algorithm_comparison.png"
        )
    
    print("\n" + "="*70)
    print("✅ Visualization Complete!")
    print("="*70)
    print(f"📁 All plots saved to: results/plots/")
    
    # 生成されたファイル一覧
    plot_files = os.listdir('results/plots')
    if plot_files:
        print("\n📋 Generated files:")
        for file in sorted(plot_files):
            print(f"   - {file}")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()