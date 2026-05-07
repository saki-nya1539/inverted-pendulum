"""
強化学習エージェントの評価
"""

import numpy as np
from typing import Tuple, Dict, List
from .cart_pole_env import CartPoleEnv


class Evaluator:
    """
    訓練済みエージェントの評価クラス
    """
    
    def __init__(self, agent, env: CartPoleEnv):
        """
        Parameters
        ----------
        agent : Agent
            評価するエージェント
        env : CartPoleEnv
            環境
        """
        self.agent = agent
        self.env = env
        self.evaluation_rewards = []
        self.evaluation_lengths = []
    
    def evaluate_episode(self, render: bool = False) -> Tuple[float, int]:
        """
        1エピソード評価（学習なし）
        
        Parameters
        ----------
        render : bool
            環境をレンダリングするか
            
        Returns
        -------
        Tuple[float, int]
            (エピソード報酬, エピソード長)
        """
        state = self.env.reset()
        
        # Q-Learning 用に状態を離散化
        if hasattr(self.agent, 'state_bins'):
            state = self.env.discretize_state(state, bins=self.agent.state_bins)
        
        episode_reward = 0.0
        episode_length = 0
        
        while True:
            # アクション選択（学習モード=False）
            action = self.agent.select_action(state, training=False)
            
            # 環境のステップ
            next_state, reward, done, info = self.env.step(action)
            
            # Q-Learning 用に状態を離散化
            if hasattr(self.agent, 'state_bins'):
                next_state = self.env.discretize_state(next_state, bins=self.agent.state_bins)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if render:
                self.env.render()
            
            if done or episode_length >= 500:
                break
        
        return episode_reward, episode_length
    
    def evaluate(self, num_episodes: int = 10, render: bool = False) -> Dict[str, float]:
        """
        複数エピソード評価
        
        Parameters
        ----------
        num_episodes : int
            評価エピソード数
        render : bool
            環境をレンダリングするか
            
        Returns
        -------
        Dict[str, float]
            評価結果統計
        """
        print(f"\n{'='*60}")
        print(f"Evaluating {self.agent.__class__.__name__}")
        print(f"{'='*60}")
        print(f"Evaluation Episodes: {num_episodes}\n")
        
        self.evaluation_rewards = []
        self.evaluation_lengths = []
        
        for episode in range(1, num_episodes + 1):
            reward, length = self.evaluate_episode(render=render)
            self.evaluation_rewards.append(reward)
            self.evaluation_lengths.append(length)
            
            print(f"Episode {episode:2d} | "
                  f"Reward: {reward:6.1f} | "
                  f"Length: {length:3d}")
        
        # 統計を計算
        stats = {
            'mean_reward': np.mean(self.evaluation_rewards),
            'std_reward': np.std(self.evaluation_rewards),
            'max_reward': np.max(self.evaluation_rewards),
            'min_reward': np.min(self.evaluation_rewards),
            'mean_length': np.mean(self.evaluation_lengths),
            'std_length': np.std(self.evaluation_lengths),
            'max_length': np.max(self.evaluation_lengths),
            'min_length': np.min(self.evaluation_lengths),
        }
        
        print(f"\n{'='*60}")
        print(f"Evaluation Results")
        print(f"{'='*60}")
        print(f"Mean Reward:  {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Max Reward:   {stats['max_reward']:.2f}")
        print(f"Min Reward:   {stats['min_reward']:.2f}")
        print(f"Mean Length:  {stats['mean_length']:.2f} ± {stats['std_length']:.2f}")
        print(f"Max Length:   {stats['max_length']:.0f}")
        print(f"Min Length:   {stats['min_length']:.0f}")
        print(f"{'='*60}\n")
        
        return stats
    
    def get_success_rate(self, threshold: float = 195.0) -> float:
        """
        成功率を計算（報酬が閾値以上のエピソード率）
        
        Parameters
        ----------
        threshold : float
            成功の閾値
            
        Returns
        -------
        float
            成功率 (0.0 ~ 1.0)
        """
        if not self.evaluation_rewards:
            return 0.0
        
        successes = sum(1 for r in self.evaluation_rewards if r >= threshold)
        return successes / len(self.evaluation_rewards)
    
    def compare_agents(self, agents_dict: Dict, num_episodes: int = 10) -> Dict:
        """
        複数のエージェントを比較評価
        
        Parameters
        ----------
        agents_dict : Dict
            {エージェント名: エージェントオブジェクト} の辞書
        num_episodes : int
            各エージェントの評価エピソード数
            
        Returns
        -------
        Dict
            比較結果
        """
        print(f"\n{'='*60}")
        print(f"Agent Comparison")
        print(f"{'='*60}\n")
        
        results = {}
        
        for agent_name, agent in agents_dict.items():
            self.agent = agent
            stats = self.evaluate(num_episodes=num_episodes, render=False)
            results[agent_name] = stats
            print()
        
        # 比較表を表示
        print(f"\n{'='*60}")
        print(f"Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Agent':<20} {'Mean Reward':<15} {'Max Reward':<15}")
        print(f"{'-'*50}")
        
        for agent_name, stats in results.items():
            print(f"{agent_name:<20} {stats['mean_reward']:<15.2f} {stats['max_reward']:<15.2f}")
        
        print(f"{'='*60}\n")
        
        return results