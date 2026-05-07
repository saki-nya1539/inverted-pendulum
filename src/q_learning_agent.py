"""
Q-Learning エージェント
"""

import numpy as np
import pickle
from typing import Tuple
from .agent import Agent


class QLearningAgent(Agent):
    """
    Q-Learning エージェント
    
    テーブル型の強化学習アルゴリズム
    """
    
    def __init__(self, num_states: int = 20,
                 num_actions: int = 2,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 state_bins: int = 10):
        """
        Parameters
        ----------
        num_states : int
            状態数（またはビン数）
        num_actions : int
            アクション数
        learning_rate : float
            学習率（α）
        discount_factor : float
            割引因子（γ）
        epsilon : float
            初期ε-greedy 探索パラメータ
        epsilon_decay : float
            各エピソード終了時のε減衰率
        epsilon_min : float
            εの最小値
        state_bins : int
            各状態次元のビン数
        """
        super().__init__(num_actions, learning_rate)
        
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.state_bins = state_bins
        
        # Q-table の初期化：NumPy配列版（テスト互換性）
        self.q_table = np.zeros((num_states, num_actions))
        
        # ディクショナリ版：柔軟な状態表現用
        self.q_table_dict = {}
    
    def _get_state_key(self, state: Tuple) -> Tuple:
        """状態をQ-tableのキーに変換"""
        return tuple(state) if isinstance(state, (list, np.ndarray)) else state
    
    def _ensure_state_exists(self, state_key: Tuple):
        """状態がQ-tableに存在することを保証"""
        if state_key not in self.q_table_dict:
            self.q_table_dict[state_key] = np.zeros(self.num_actions)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy 戦略でアクションを選択
        
        Parameters
        ----------
        state : np.ndarray
            現在の状態
        training : bool
            訓練モード
            
        Returns
        -------
        int
            選択されたアクション
        """
        # 状態がタプルまたはインデックスかを判定
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], int):
            # タプル形式：(state_idx, action_idx)
            state_key = state
            self._ensure_state_exists(state_key)
            
            if training and np.random.rand() < self.epsilon:
                return np.random.randint(self.num_actions)
            else:
                q_values = self.q_table_dict[state_key]
                return np.argmax(q_values)
        else:
            # NumPy配列形式
            if training and np.random.rand() < self.epsilon:
                return np.random.randint(self.num_actions)
            else:
                q_values = self.q_table[int(state[0]) % self.num_states]
                return np.argmax(q_values)
    
    def update_q_value(self, state: np.ndarray, action: int, reward: float,
                       next_state: np.ndarray, done: bool) -> float:
        """
        Q値を更新
        
        Parameters
        ----------
        state : np.ndarray
            現在の状態
        action : int
            実行したアクション
        reward : float
            報酬
        next_state : np.ndarray
            次の状態
        done : bool
            エピソード終了フラグ
            
        Returns
        -------
        float
            TD誤差
        """
        # 状態がタプル形式の場合、NumPy配列へ変換
        if isinstance(state, tuple):
            # タプルから直接アクセスできるように修正
            state_idx = state[0]
            next_state_idx = next_state[0]
            
            # Q値の現在値
            current_q = self.q_table[state_idx, action]
            
            # 次の状態での最大Q値
            if done:
                max_next_q = 0.0
            else:
                max_next_q = np.max(self.q_table[next_state_idx])
            
            # TD目標
            td_target = reward + self.discount_factor * max_next_q
            
            # TD誤差
            td_error = td_target - current_q
            
            # Q値を更新
            self.q_table[state_idx, action] += self.learning_rate * td_error
            
            return abs(td_error)
        else:
            # learn()メソッドに委譲
            return self.learn(state, action, reward, next_state, done)
    
    def learn(self, state: np.ndarray, action: int, reward: float,
              next_state: np.ndarray, done: bool) -> float:
        """
        Q-Learning の学習ステップ
        
        Q(s,a) ← Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        
        Parameters
        ----------
        state : np.ndarray
            現在の状態
        action : int
            実行したアクション
        reward : float
            報酬
        next_state : np.ndarray
            次の状態
        done : bool
            エピソード終了フラグ
            
        Returns
        -------
        float
            TD誤差
        """
        # 状態がタプル形式か判定
        if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], int):
            state_key = state
            next_state_key = next_state
            
            self._ensure_state_exists(state_key)
            self._ensure_state_exists(next_state_key)
            
            # Q値の現在値
            current_q = self.q_table_dict[state_key][action]
            
            # 次の状態での最大Q値
            if done:
                max_next_q = 0.0
            else:
                max_next_q = np.max(self.q_table_dict[next_state_key])
            
            # TD目標
            td_target = reward + self.discount_factor * max_next_q
            
            # TD誤差
            td_error = td_target - current_q
            
            # Q値を更新
            self.q_table_dict[state_key][action] += self.learning_rate * td_error
            
            return abs(td_error)
        else:
            # NumPy配列形式
            state_idx = int(state[0]) % self.num_states
            next_state_idx = int(next_state[0]) % self.num_states
            
            # Q値の現在値
            current_q = self.q_table[state_idx, action]
            
            # 次の状態での最大Q値
            if done:
                max_next_q = 0.0
            else:
                max_next_q = np.max(self.q_table[next_state_idx])
            
            # TD目標
            td_target = reward + self.discount_factor * max_next_q
            
            # TD誤差
            td_error = td_target - current_q
            
            # Q値を更新
            self.q_table[state_idx, action] += self.learning_rate * td_error
            
            return abs(td_error)
    
    def decay_epsilon(self):
        """ε を減衰（エピソード終了時に呼び出し）"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def decay_learning_rate(self):
        """学習率を減衰"""
        self.learning_rate *= 0.9999
    
    def save_model(self, filepath: str):
        """
        Q-table をピクルファイルに保存
        
        Parameters
        ----------
        filepath : str
            保存先パス
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'q_table_dict': self.q_table_dict}, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Q-table をピクルファイルから読み込み
        
        Parameters
        ----------
        filepath : str
            読み込み元パス
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.q_table_dict = data['q_table_dict']
        print(f"Model loaded from {filepath}")
    
    def get_q_table_size(self) -> int:
        """Q-tableのサイズを取得"""
        return len(self.q_table_dict)