"""
Q-Learning エージェントのテスト
"""

import unittest
import numpy as np
from src.cart_pole_env import CartPoleEnv
from src.q_learning_agent import QLearningAgent


class TestQLearningAgent(unittest.TestCase):
    """
    QLearningAgent のテストクラス
    """
    
    def setUp(self):
        """各テストの前に実行"""
        self.env = CartPoleEnv()
        self.agent = QLearningAgent(
            num_states=20,
            num_actions=2,
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
    
    def tearDown(self):
        """各テストの後に実行"""
        self.env.close()
    
    def test_agent_initialization(self):
        """エージェント初期化テスト"""
        self.assertEqual(self.agent.num_states, 20)
        self.assertEqual(self.agent.num_actions, 2)
        self.assertEqual(self.agent.learning_rate, 0.1)
        self.assertEqual(self.agent.discount_factor, 0.99)
    
    def test_q_table_creation(self):
        """Q-テーブル作成テスト"""
        self.assertIsNotNone(self.agent.q_table)
        self.assertEqual(self.agent.q_table.shape, (20, 2))
    
    def test_select_action_training(self):
        """訓練モードでのアクション選択テスト"""
        state = (5, 3)
        
        # 複数回アクション選択
        actions = []
        for _ in range(100):
            action = self.agent.select_action(state, training=True)
            actions.append(action)
            
            # アクションが0または1か確認
            self.assertIn(action, [0, 1])
        
        # 訓練モードなので様々なアクションが選択されるはず
        self.assertGreater(len(set(actions)), 1)
    
    def test_select_action_evaluation(self):
        """評価モードでのアクション選択テスト"""
        state = (5, 3)
        
        # Q-テーブルの値を設定
        self.agent.q_table[5, 0] = 10.0
        self.agent.q_table[5, 1] = 5.0
        
        # 評価モードではQ値が高いアクションを選択
        action = self.agent.select_action(state, training=False)
        self.assertEqual(action, 0)
    
    def test_update_q_value(self):
        """Q値更新テスト"""
        state = (5, 3)
        action = 0
        reward = 1.0
        next_state = (6, 4)
        done = False
        
        # 更新前のQ値
        q_old = self.agent.q_table[state[0], action]
        
        # Q値を更新
        self.agent.update_q_value(state, action, reward, next_state, done)
        
        # Q値が変わったか確認
        q_new = self.agent.q_table[state[0], action]
        self.assertNotEqual(q_old, q_new)
    
    def test_epsilon_decay(self):
        """εの減衰テスト"""
        initial_epsilon = self.agent.epsilon
        
        # 100回減衰
        for _ in range(100):
            self.agent.decay_epsilon()
        
        # εが減少したか確認
        self.assertLess(self.agent.epsilon, initial_epsilon)
        
        # εが最小値以上か確認
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)
    
    def test_learning_rate_decay(self):
        """学習率の減衰テスト"""
        initial_lr = self.agent.learning_rate
        
        # 100回減衰
        for _ in range(100):
            self.agent.decay_learning_rate()
        
        # 学習率が減少したか確認
        self.assertLess(self.agent.learning_rate, initial_lr)
    
    def test_save_and_load_model(self):
        """モデルの保存と読み込みテスト"""
        import tempfile
        import os
        
        # 一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            # Q-テーブルを設定
            self.agent.q_table[0, 0] = 99.9
            
            # モデルを保存
            self.agent.save_model(temp_path)
            
            # 新しいエージェントを作成
            agent2 = QLearningAgent(
                num_states=20,
                num_actions=2,
                learning_rate=0.1,
                discount_factor=0.99
            )
            
            # モデルを読み込み
            agent2.load_model(temp_path)
            
            # Q-テーブルが一致するか確認
            np.testing.assert_array_equal(self.agent.q_table, agent2.q_table)
        
        finally:
            # 一時ファイルを削除
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_one_episode_training(self):
        """1エピソード訓練テスト"""
        state = self.env.reset()
        state = self.env.discretize_state(state, bins=self.agent.state_bins)
        
        episode_reward = 0
        
        for step in range(100):
            action = self.agent.select_action(state, training=True)
            next_state, reward, done, _ = self.env.step(action)
            next_state = self.env.discretize_state(next_state, bins=self.agent.state_bins)
            
            self.agent.update_q_value(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # エピソード報酬が正の値か確認
        self.assertGreater(episode_reward, 0)


if __name__ == '__main__':
    unittest.main()