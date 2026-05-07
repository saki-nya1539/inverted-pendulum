"""
強化学習エージェント基底クラス
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple


class Agent(ABC):
    """
    RL エージェントの基底クラス
    """
    
    def __init__(self, action_space_size: int, learning_rate: float = 0.01):
        """
        Parameters
        ----------
        action_space_size : int
            アクション空間のサイズ
        learning_rate : float
            学習率
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
    
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        状態に基づいてアクションを選択
        
        Parameters
        ----------
        state : np.ndarray
            現在の状態
        training : bool
            訓練モード（True）か評価モード（False）か
            
        Returns
        -------
        int
            選択されたアクション
        """
        pass
    
    @abstractmethod
    def learn(self, state: np.ndarray, action: int, reward: float, 
              next_state: np.ndarray, done: bool) -> float:
        """
        学習ステップ
        
        Parameters
        ----------
        state : np.ndarray
            現在の状態
        action : int
            実行したアクション
        reward : float
            獲得した報酬
        next_state : np.ndarray
            次の状態
        done : bool
            エピソード終了フラグ
            
        Returns
        -------
        float
            この学習ステップのロス/TD誤差
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str):
        """
        モデルを保存
        
        Parameters
        ----------
        filepath : str
            保存先パス
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str):
        """
        モデルを読み込み
        
        Parameters
        ----------
        filepath : str
            読み込み元パス
        """
        pass