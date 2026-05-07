"""
OpenAI Gym Cart-Pole Environment Wrapper
"""

import gym
import numpy as np
from typing import Tuple, Dict, Any


class CartPoleEnv:
    """
    Cart-Pole 環境のラッパークラス
    
    OpenAI Gym の CartPole-v1 環境をラップして
    統一されたインターフェースを提供
    """
    
    def __init__(self, render_mode: str = None):
        """
        Parameters
        ----------
        render_mode : str, optional
            レンダリングモード ('human', 'rgb_array', None)
        """
        self.env = gym.make('CartPole-v1', render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # 状態の正規化用
        self.state_mean = np.array([0.0, 0.0, 0.0, 0.0])
        self.state_std = np.array([2.4, 0.2, 2.0, 2.0])
    
    def reset(self) -> np.ndarray:
        """
        環境をリセット
        
        Returns
        -------
        np.ndarray
            初期状態
        """
        state, info = self.env.reset()
        return np.array(state, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        1ステップシミュレーション
        
        Parameters
        ----------
        action : int
            実行するアクション (0: 左, 1: 右)
            
        Returns
        -------
        Tuple
            (状態, 報酬, 終了フラグ, 切断フラグ, 情報)
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        state = np.array(state, dtype=np.float32)
        
        done = terminated or truncated
        
        return state, float(reward), done, info
    
    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        状態を正規化
        
        Parameters
        ----------
        state : np.ndarray
            元の状態
            
        Returns
        -------
        np.ndarray
            正規化された状態
        """
        return (state - self.state_mean) / self.state_std
    
    def discretize_state(self, state: np.ndarray, bins: int = 10) -> Tuple:
        """
        連続状態を離散化（Q-Learning用）
        
        Parameters
        ----------
        state : np.ndarray
            連続状態
        bins : int
            各次元のビン数
            
        Returns
        -------
        Tuple
            離散化された状態
        """
        # 各次元を正規化してビンに変換
        discretized = []
        for i, value in enumerate(state):
            normalized = (value - self.state_mean[i]) / self.state_std[i]
            # [-1, 1] に正規化
            normalized = np.clip(normalized, -1.0, 1.0)
            # ビンインデックスに変換
            bin_index = int((normalized + 1.0) / 2.0 * (bins - 1))
            discretized.append(bin_index)
        
        return tuple(discretized)
    
    def render(self):
        """環境をレンダリング"""
        self.env.render()
    
    def close(self):
        """環境をクローズ"""
        self.env.close()