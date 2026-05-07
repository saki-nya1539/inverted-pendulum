"""
訓練結果の可視化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Dict, Tuple
from .cart_pole_env import CartPoleEnv


class Visualizer:
    """
    訓練結果とエージェント動作を可視化するクラス
    """
    
    def __init__(self, save_path: str = 'results/plots'):
        """
        Parameters
        ----------
        save_path : str
            図を保存するパス
        """
        self.save_path = save_path
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_training_history(self, 
                             rewards: List[float], 
                             title: str = "Training History",
                             filename: str = "training_history.png"):
        """
        訓練の報酬履歴をプロット
        
        Parameters
        ----------
        rewards : List[float]
            エピソード毎の報酬
        title : str
            グラフのタイトル
        filename : str
            保存ファイル名
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 報酬をプロット
        ax.plot(rewards, label='Episode Reward', alpha=0.6, linewidth=1)
        
        # 移動平均をプロット
        window = 100
        if len(rewards) >= window:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 
                   label=f'{window}-Episode Moving Average', 
                   linewidth=2, color='red')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{filename}', dpi=300)
        print(f"✅ Saved: {self.save_path}/{filename}")
        plt.close()
    
    def plot_comparison(self, 
                       data_dict: Dict[str, List[float]], 
                       title: str = "Agent Comparison",
                       filename: str = "comparison.png"):
        """
        複数のエージェントを比較するグラフ
        
        Parameters
        ----------
        data_dict : Dict[str, List[float]]
            {エージェント名: 報酬リスト} の辞書
        title : str
            グラフのタイトル
        filename : str
            保存ファイル名
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data_dict)))
        
        for (agent_name, rewards), color in zip(data_dict.items(), colors):
            # 移動平均を計算
            window = 100
            if len(rewards) >= window:
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(rewards)), moving_avg, 
                       label=agent_name, linewidth=2, color=color)
            else:
                ax.plot(rewards, label=agent_name, linewidth=2, color=color)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward (Moving Average)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{filename}', dpi=300)
        print(f"✅ Saved: {self.save_path}/{filename}")
        plt.close()
    
    def plot_statistics(self, 
                       stats_dict: Dict[str, Dict[str, float]],
                       filename: str = "statistics.png"):
        """
        複数のエージェントの統計を比較
        
        Parameters
        ----------
        stats_dict : Dict[str, Dict[str, float]]
            {エージェント名: {統計名: 値}} の辞書
        filename : str
            保存ファイル名
        """
        agent_names = list(stats_dict.keys())
        mean_rewards = [stats_dict[name]['mean_reward'] for name in agent_names]
        std_rewards = [stats_dict[name]['std_reward'] for name in agent_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 平均報酬
        colors = plt.cm.Set2(np.linspace(0, 1, len(agent_names)))
        bars1 = ax1.bar(agent_names, mean_rewards, color=colors, alpha=0.7, edgecolor='black')
        ax1.errorbar(agent_names, mean_rewards, yerr=std_rewards, fmt='none', 
                    color='black', capsize=5, capthick=2)
        ax1.set_ylabel('Mean Reward', fontsize=12)
        ax1.set_title('Mean Reward Comparison', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 値をバーの上に表示
        for bar, mean, std in zip(bars1, mean_rewards, std_rewards):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mean:.1f}±{std:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        # 最大報酬
        max_rewards = [stats_dict[name]['max_reward'] for name in agent_names]
        bars2 = ax2.bar(agent_names, max_rewards, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Max Reward', fontsize=12)
        ax2.set_title('Max Reward Comparison', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 値をバーの上に表示
        for bar, max_r in zip(bars2, max_rewards):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{max_r:.1f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{filename}', dpi=300)
        print(f"✅ Saved: {self.save_path}/{filename}")
        plt.close()
    
    def visualize_episode(self, 
                         agent, 
                         env: CartPoleEnv,
                         max_steps: int = 500,
                         filename: str = "episode.gif"):
        """
        エピソード中のエージェント動作を可視化
        
        Parameters
        ----------
        agent : Agent
            エージェント
        env : CartPoleEnv
            環境
        max_steps : int
            最大ステップ数
        filename : str
            保存ファイル名
        """
        print(f"\n📹 Recording episode...")
        
        frames = []
        state = env.reset()
        
        # Q-Learning 用に状態を離散化
        if hasattr(agent, 'state_bins'):
            state = env.discretize_state(state, bins=agent.state_bins)
        
        for step in range(max_steps):
            # フレームを取得
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            
            # アクション選択
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            # Q-Learning 用に状態を離散化
            if hasattr(agent, 'state_bins'):
                next_state = env.discretize_state(next_state, bins=agent.state_bins)
            
            state = next_state
            
            if done:
                break
        
        # GIF に保存
        if frames:
            self._save_gif(frames, filename)
    
    def _save_gif(self, frames: List[np.ndarray], filename: str):
        """
        フレームリストをGIFに保存
        
        Parameters
        ----------
        frames : List[np.ndarray]
            フレームのリスト
        filename : str
            保存ファイル名
        """
        import imageio
        
        output_path = f'{self.save_path}/{filename}'
        imageio.mimsave(output_path, frames, fps=30)
        print(f"✅ Saved: {output_path}")
    
    def plot_q_table_heatmap(self, 
                            q_table: np.ndarray,
                            filename: str = "q_table_heatmap.png"):
        """
        Q-テーブルをヒートマップで可視化
        
        Parameters
        ----------
        q_table : np.ndarray
            Q-テーブル
        filename : str
            保存ファイル名
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(q_table, cmap='viridis', aspect='auto')
        
        ax.set_xlabel('Action', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        ax.set_title('Q-Table Heatmap', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Q-Value', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{filename}', dpi=300)
        print(f"✅ Saved: {self.save_path}/{filename}")
        plt.close()
    
    def plot_episode_length_distribution(self,
                                        episode_lengths: List[int],
                                        title: str = "Episode Length Distribution",
                                        filename: str = "episode_lengths.png"):
        """
        エピソード長の分布をプロット
        
        Parameters
        ----------
        episode_lengths : List[int]
            各エピソードの長さ
        title : str
            グラフのタイトル
        filename : str
            保存ファイル名
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(episode_lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(episode_lengths), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(episode_lengths):.1f}')
        ax.axvline(np.median(episode_lengths), color='green', linestyle='--', 
                  linewidth=2, label=f'Median: {np.median(episode_lengths):.1f}')
        
        ax.set_xlabel('Episode Length', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{filename}', dpi=300)
        print(f"✅ Saved: {self.save_path}/{filename}")
        plt.close()
    
    def plot_cartpole_state(self,
                           states: List[np.ndarray],
                           filename: str = "cartpole_states.png"):
        """
        Cart-Pole の状態変化を可視化
        
        Parameters
        ----------
        states : List[np.ndarray]
            各タイムステップの状態 [x, x_dot, theta, theta_dot]
        filename : str
            保存ファイル名
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        states = np.array(states)
        state_names = ['Cart Position (x)', 'Cart Velocity (ẋ)', 
                      'Pole Angle (θ)', 'Pole Angular Velocity (θ̇)']
        
        for idx, (ax, state_name) in enumerate(zip(axes.flat, state_names)):
            ax.plot(states[:, idx], linewidth=2, color='blue')
            ax.set_xlabel('Time Step', fontsize=11)
            ax.set_ylabel(state_name, fontsize=11)
            ax.set_title(state_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_path}/{filename}', dpi=300)
        print(f"✅ Saved: {self.save_path}/{filename}")
        plt.close()