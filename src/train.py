"""
強化学習エージェントの訓練スクリプト
"""

import numpy as np
import pickle
import os
import torch
from typing import Tuple, List, Dict
from .cart_pole_env import CartPoleEnv
from .q_learning_agent import QLearningAgent
from .dqn_agent import DQNAgent
from .evaluate import Evaluator
from .visualization import Visualizer


def train_q_learning(env: CartPoleEnv,
                     agent: QLearningAgent,
                     num_episodes: int = 1000,
                     render: bool = False) -> Tuple[List[float], List[int]]:
    """
    Q-Learning アルゴリズムで訓練

    Parameters
    ----------
    env : CartPoleEnv
        環境
    agent : QLearningAgent
        Q-Learning エージェント
    num_episodes : int
        訓練エピソード数
    render : bool
        環境をレンダリングするか

    Returns
    -------
    Tuple[List[float], List[int]]
        (報酬リスト, エピソード長リスト)
    """
    print(f"\n{'='*60}")
    print(f"Training Q-Learning Agent")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"State Bins: {agent.state_bins}")
    print(f"Learning Rate: {agent.learning_rate}")
    print(f"Discount Factor: {agent.discount_factor}")
    print(f"Epsilon Start: {agent.epsilon}")
    print(f"{'='*60}\n")

    rewards = []
    episode_lengths = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        state = env.discretize_state(state, bins=agent.state_bins)

        episode_reward = 0.0
        episode_length = 0

        while True:
            # アクション選択（ε-greedy）
            action = agent.select_action(state, training=True)

            # 環境のステップ
            next_state, reward, done, info = env.step(action)
            next_state = env.discretize_state(next_state, bins=agent.state_bins)

            # Q値を更新
            agent.update_q_value(state, action, reward, next_state, done)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if render:
                env.render()

            if done or episode_length >= 500:
                break

        # 学習率とεを減衰
        agent.decay_learning_rate()
        agent.decay_epsilon()

        rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # 100エピソード毎に進捗を表示
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Average Reward: {np.mean(rewards[-100:]):.2f}")
    print(f"{'='*60}\n")

    return rewards, episode_lengths


def train_dqn(env: CartPoleEnv,
              agent: DQNAgent,
              num_episodes: int = 500,
              render: bool = False) -> Tuple[List[float], List[int], List[float]]:
    """
    Deep Q-Network (DQN) で訓練

    Parameters
    ----------
    env : CartPoleEnv
        環境
    agent : DQNAgent
        DQN エージェント
    num_episodes : int
        訓練エピソード数
    render : bool
        環境をレンダリングするか

    Returns
    -------
    Tuple[List[float], List[int], List[float]]
        (報酬リスト, エピソード長リスト, ロスリスト)
    """
    print(f"\n{'='*60}")
    print(f"Training DQN Agent")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"Learning Rate: {agent.learning_rate}")
    print(f"Discount Factor: {agent.discount_factor}")
    print(f"Epsilon Start: {agent.epsilon}")
    print(f"Batch Size: {agent.batch_size}")
    print(f"{'='*60}\n")

    rewards = []
    episode_lengths = []
    losses = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()

        episode_reward = 0.0
        episode_length = 0

        while True:
            # アクション選択（ε-greedy）
            action = agent.select_action(state, training=True)

            # 環境のステップ
            next_state, reward, done, info = env.step(action)

            # リプレイバッファに追加
            agent.remember(state, action, reward, next_state, done)

            episode_reward += reward
            episode_length += 1
            state = next_state

            if render:
                env.render()

            if done or episode_length >= 500:
                break

        # ネットワークを訓練
        if len(agent.memory) >= agent.batch_size:
            loss = agent.train_on_batch()
            losses.append(loss)

        # εを減衰
        agent.decay_epsilon()

        rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # ターゲットネットワークを更新
        if episode % 10 == 0:
            agent.update_target_network()

        # 100エピソード毎に進捗を表示
        if episode % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_loss = np.mean(losses[-100:]) if losses else 0.0
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Avg Length: {avg_length:5.1f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Average Reward: {np.mean(rewards[-100:]):.2f}")
    print(f"{'='*60}\n")

    return rewards, episode_lengths, losses


def save_training_results(rewards: List[float],
                         episode_lengths: List[int],
                         agent,
                         save_dir: str = "data/models"):
    """
    訓練結果とモデルを保存

    Parameters
    ----------
    rewards : List[float]
        エピソード報酬
    episode_lengths : List[int]
        エピソード長
    agent : Agent
        訓練済みエージェント
    save_dir : str
        保存ディレクトリ
    """
    # ディレクトリを作成
    os.makedirs(save_dir, exist_ok=True)

    agent_name = agent.__class__.__name__
    model_path = os.path.join(save_dir, f"{agent_name.lower()}_model.pkl")

    try:
        # エージェントオブジェクト全体を pickle で保存（推奨）
        with open(model_path, 'wb') as f:
            pickle.dump(agent, f)
        print(f"✅ Model saved: {model_path}")
    except Exception as e:
        print(f"⚠️  Failed to save with pickle, trying alternative method: {e}")
        try:
            # DQN の場合は state_dict で保存
            if hasattr(agent, 'model'):
                torch.save(agent.model.state_dict(), model_path.replace('.pkl', '.pth'))
                print(f"✅ Model saved (PyTorch): {model_path.replace('.pkl', '.pth')}")
        except Exception as e2:
            print(f"❌ Failed to save model: {e2}")

    # 訓練履歴を保存
    history_path = os.path.join(save_dir, f"{agent_name.lower()}_history.pkl")
    history = {
        'rewards': rewards,
        'episode_lengths': episode_lengths
    }
    try:
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        print(f"✅ History saved: {history_path}")
    except Exception as e:
        print(f"❌ Failed to save history: {e}")


def load_training_results(agent_name: str,
                         load_dir: str = "data/models"):
    """
    保存した訓練結果を読み込み

    Parameters
    ----------
    agent_name : str
        エージェント名 ("QLearningAgent" または "DQNAgent")
    load_dir : str
        読み込みディレクトリ

    Returns
    -------
    Tuple[List[float], List[int]]
        (報酬リスト, エピソード長リスト)
    """
    history_path = os.path.join(load_dir, f"{agent_name.lower()}_history.pkl")

    if os.path.exists(history_path):
        try:
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            print(f"✅ History loaded: {history_path}")
            return history['rewards'], history['episode_lengths']
        except Exception as e:
            print(f"❌ Failed to load history: {e}")
            return [], []
    else:
        print(f"❌ History file not found: {history_path}")
        return [], []


def main_q_learning():
    """
    Q-Learning の訓練メイン関数
    """
    # 環境とエージェントを作成
    env = CartPoleEnv()
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
    rewards, episode_lengths = train_q_learning(
        env=env,
        agent=agent,
        num_episodes=1000,
        render=False
    )

    # 結果を保存
    save_training_results(rewards, episode_lengths, agent)

    # 評価
    evaluator = Evaluator(agent, env)
    stats = evaluator.evaluate(num_episodes=10)

    # 可視化
    viz = Visualizer(save_path='results/plots')
    os.makedirs('results/plots', exist_ok=True)
    viz.plot_training_history(rewards, title="Q-Learning Training")

    env.close()


def main_dqn():
    """
    DQN の訓練メイン関数
    """
    # 環境とエージェントを作成
    env = CartPoleEnv()
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
    rewards, episode_lengths, losses = train_dqn(
        env=env,
        agent=agent,
        num_episodes=500,
        render=False
    )

    # 結果を保存
    save_training_results(rewards, episode_lengths, agent)

    # 評価
    evaluator = Evaluator(agent, env)
    stats = evaluator.evaluate(num_episodes=10)

    # 可視化
    viz = Visualizer(save_path='results/plots')
    os.makedirs('results/plots', exist_ok=True)
    viz.plot_training_history(rewards, title="DQN Training")

    env.close()


if __name__ == "__main__":
    # Q-Learning を実行
    print("\n" + "="*60)
    print("Q-LEARNING TRAINING")
    print("="*60)
    main_q_learning()

    # DQN を実行
    print("\n" + "="*60)
    print("DQN TRAINING")
    print("="*60)
    main_dqn()