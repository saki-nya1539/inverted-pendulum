"""
Deep Q-Network (DQN) エージェント (PyTorch版)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple
from .agent import Agent


class DQNNetwork(nn.Module):
    """DQNニューラルネットワーク"""
    
    def __init__(self, state_size: int, action_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        self.relu = nn.ReLU()
    
    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def get_weights(self):
        """Keras互換：全パラメータをリストで取得"""
        return [p.data.clone() for p in self.parameters()]
    
    def set_weights(self, weights):
        """Keras互換：全パラメータを設定"""
        for p, w in zip(self.parameters(), weights):
            p.data.copy_(w)


class DQNAgent(Agent):
    """
    Deep Q-Network (DQN) エージェント
    
    ニューラルネットワークを使用した深層強化学習
    """
    
    def __init__(self, state_size: int = 4,
                 action_size: int = 2,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        """
        Parameters
        ----------
        state_size : int
            状態空間のサイズ
        action_size : int
            アクション数
        learning_rate : float
            学習率
        discount_factor : float
            割引因子（γ）
        epsilon : float
            初期ε
        epsilon_decay : float
            εの減衰率
        epsilon_min : float
            εの最小値
        memory_size : int
            経験リプレイのメモリサイズ
        batch_size : int
            バッチサイズ
        """
        super().__init__(action_size, learning_rate)
        
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 経験リプレイメモリ
        self.memory_storage = {
            'states': np.zeros((memory_size, state_size), dtype=np.float32),
            'actions': np.zeros(memory_size, dtype=np.int32),
            'rewards': np.zeros(memory_size, dtype=np.float32),
            'next_states': np.zeros((memory_size, state_size), dtype=np.float32),
            'dones': np.zeros(memory_size, dtype=bool),
        }
        self.memory = []  # テスト用：len(agent.memory) をサポート
        self.memory_size = memory_size
        self.memory_index = 0
        
        # ニューラルネットワークの構築
        self.model = DQNNetwork(state_size, action_size).to(self.device)
        self.target_model = DQNNetwork(state_size, action_size).to(self.device)
        self.update_target_network()
        
        # 最適化器と損失関数
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
    
    def update_target_network(self):
        """ターゲットネットワークをメインネットワークにコピー"""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                 next_state: np.ndarray, done: bool):
        """
        経験をメモリに保存
        
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
        """
        index = self.memory_index % self.memory_size
        self.memory_storage['states'][index] = state
        self.memory_storage['actions'][index] = action
        self.memory_storage['rewards'][index] = reward
        self.memory_storage['next_states'][index] = next_state
        self.memory_storage['dones'][index] = done
        
        # テスト用：リストにも追加
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        self.memory.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
        
        self.memory_index += 1
    
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
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        
        # Q値を予測
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax(dim=1).item()
    
    def learn(self, state: np.ndarray = None, action: int = None,
              reward: float = None, next_state: np.ndarray = None,
              done: bool = None) -> float:
        """
        経験リプレイによる学習
        
        Returns
        -------
        float
            学習ロス
        """
        if self.memory_index < self.batch_size:
            return 0.0
        
        # ミニバッチをランダムにサンプリング
        indices = np.random.choice(min(self.memory_index, self.memory_size), 
                                  self.batch_size, replace=False)
        
        states = torch.FloatTensor(self.memory_storage['states'][indices]).to(self.device)
        actions = torch.LongTensor(self.memory_storage['actions'][indices]).to(self.device)
        rewards = torch.FloatTensor(self.memory_storage['rewards'][indices]).to(self.device)
        next_states = torch.FloatTensor(self.memory_storage['next_states'][indices]).to(self.device)
        dones = torch.BoolTensor(self.memory_storage['dones'][indices]).to(self.device)
        
        # 現在のQ値を計算
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        
        # ターゲットQ値を計算
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + self.discount_factor * next_q_values * (~dones).float()
        
        # 損失を計算
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # バックプロパゲーション
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_on_batch(self) -> float:
        """
        テスト互換のエイリアス
        経験リプレイからミニバッチを取り出して学習を行う
        
        Returns
        -------
        float
            学習ロス
        """
        return self.learn()
    
    def decay_epsilon(self):
        """ε を減衰"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save_model(self, filepath: str):
        """
        モデルを保存
        
        Parameters
        ----------
        filepath : str
            保存先パス
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        モデルを読み込み
        
        Parameters
        ----------
        filepath : str
            読み込み元パス
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        self.update_target_network()
        print(f"Model loaded from {filepath}")