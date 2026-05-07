"""
複数のエージェントを評価・比較するスクリプト
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cart_pole_env import CartPoleEnv
from src.q_learning_agent import QLearningAgent
from src.dqn_agent import DQNAgent
from src.evaluate import Evaluator
from src.visualization import Visualizer
import pickle
import traceback


def load_agent(agent_type: str, model_path: str):
    """
    保存されたモデルを読み込む

    Parameters
    ----------
    agent_type : str
        "Q-Learning" または "DQN"
    model_path : str
        モデルファイルパス

    Returns
    -------
    Agent or None
        読み込んだエージェント、失敗した場合は None
    """
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return None

    try:
        with open(model_path, 'rb') as f:
            agent = pickle.load(f)

        # 読み込んだオブジェクトの型をチェック
        print(f"   📋 Loaded object type: {type(agent).__name__}")

        # エージェントオブジェクトか確認
        if hasattr(agent, 'select_action'):
            print(f"✅ Loaded {agent_type} model from {model_path}")
            return agent
        else:
            print(f"❌ Loaded object does not have 'select_action' method")
            return None

    except pickle.UnpicklingError as e:
        print(f"❌ Unpickling error for {agent_type} model: {e}")
        traceback.print_exc()
        return None
    except Exception as e:
        print(f"❌ Failed to load {agent_type} model: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def main():
    """
    複数エージェント評価メイン処理
    """
    print("\n" + "=" * 70)
    print("📊 Agent Evaluation & Comparison")
    print("=" * 70)

    # ディレクトリ作成
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)

    # 環境作成
    print("\n📌 Creating environment...")
    env = CartPoleEnv()

    # モデルパス
    q_learning_model = 'data/models/qlearningagent_model.pkl'
    dqn_model = 'data/models/dqnagent_model.pkl'

    # エージェントを読み込み
    print("\n📌 Loading agents...")
    agents = {}

    # Q-Learning エージェント
    print("\n  Loading Q-Learning model...")
    q_agent = load_agent('Q-Learning', q_learning_model)
    if q_agent is not None:
        agents['Q-Learning'] = q_agent
    else:
        print(f"⚠️  Q-Learning agent not loaded properly")

    # DQN エージェント
    print("\n  Loading DQN model...")
    dqn_agent = load_agent('DQN', dqn_model)
    if dqn_agent is not None:
        agents['DQN'] = dqn_agent
    else:
        print(f"⚠️  DQN agent not loaded properly")

    if not agents:
        print("\n❌ No trained models found or failed to load!")
        print("\n📝 Please run training scripts first:")
        print("   python scripts/train_q_learning.py")
        print("   python scripts/train_dqn.py")
        return

    # 評価
    print("\n📌 Evaluating agents...")
    print("=" * 70)
    stats = {}

    for agent_name, agent in agents.items():
        print(f"\n🔄 Evaluating {agent_name}...")
        evaluator = Evaluator(agent, env)
        agent_stats = evaluator.evaluate(num_episodes=50)
        stats[agent_name] = agent_stats

        print(f"\n  📊 {agent_name} Results:")
        print(f"     Mean Reward: {agent_stats['mean_reward']:.2f}")
        print(f"     Max Reward:  {agent_stats['max_reward']:.2f}")
        print(f"     Min Reward:  {agent_stats['min_reward']:.2f}")
        print(f"     Std Dev:     {agent_stats['std_reward']:.2f}")

    # 可視化
    print("\n📌 Creating comparison plots...")
    try:
        viz = Visualizer(save_path='results/plots')
        viz.plot_statistics(
            stats_dict=stats,
            filename="agent_comparison.png"
        )
        print("✅ Comparison plot created successfully")
    except Exception as e:
        print(f"⚠️  Failed to create comparison plot: {e}")

    # 結果サマリー
    print("\n" + "=" * 70)
    print("📊 Comparison Summary")
    print("=" * 70)

    for agent_name, agent_stats in stats.items():
        print(f"\n🤖 {agent_name}:")
        print(f"   Mean Reward:  {agent_stats['mean_reward']:7.2f}")
        print(f"   Max Reward:   {agent_stats['max_reward']:7.2f}")
        print(f"   Min Reward:   {agent_stats['min_reward']:7.2f}")
        print(f"   Std Dev:      {agent_stats['std_reward']:7.2f}")

    # 最高スコアのエージェント
    if stats:
        best_agent = max(stats.items(), key=lambda x: x[1]['mean_reward'])
        print(f"\n🏆 Best Agent: {best_agent[0]}")
        print(f"   Mean Reward: {best_agent[1]['mean_reward']:.2f}")

        # パフォーマンス改善度
        if len(stats) > 1:
            agent_names = list(stats.keys())
            rewards = [stats[name]['mean_reward'] for name in agent_names]
            if min(rewards) > 0:
                improvement = ((max(rewards) - min(rewards)) / min(rewards)) * 100
                print(f"\n📈 Performance Improvement: {improvement:.1f}%")

    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print("=" * 70)
    print(f"📁 Comparison plot saved to: results/plots/agent_comparison.png")
    print("=" * 70 + "\n")

    env.close()


if __name__ == "__main__":
    main()