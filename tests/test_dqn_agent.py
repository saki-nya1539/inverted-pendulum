"""
DQN エージェントのテスト
"""

import unittest
import numpy as np
from src.cart_pole_env import CartPoleEnv
from src.dqn_agent import DQNAgent


class TestDQNAgent(unittest.TestCase):
    """
    DQNAgent のテストクラス
    """
    
    def setUp(self):
        """各テストの前に実行"""
        self.env = CartPoleEnv()
        self.agent = DQNAgent(
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
    
    def tearDown(self):
        """各テストの後に実行"""
        self.env.close()
    
    def test_agent_initialization(self):
        """エージェント初期化テスト"""
        self.assertEqual(self.agent.state_size, 4)
        self.assertEqual(self.agent.action_size, 2)
        self.assertEqual(self.agent.learning_rate, 0.001)
        self.assertEqual(self.agent.discount_factor, 0.99)
    
    def test_model_creation(self):
        """ニューラルネットワークモデル作成テスト"""
        self.assertIsNotNone(self.agent.model)
        self.assertIsNotNone(self.agent.target_model)
    
    def test_select_action_training(self):
        """訓練モードでのアクション選択テスト"""
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
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
        state = np.array([0.1, 0.2, 0.3, 0.4])
        
        # 評価モードでは常に同じアクションを選択
        actions = [self.agent.select_action(state, training=False) for _ in range(10)]
        
        # 全て同じアクションのはず
        self.assertEqual(len(set(actions)), 1)
    
    def test_remember(self):
        """リプレイバッファへのデータ追加テスト"""
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = 0
        reward = 1.0
        next_state = np.array([0.15, 0.25, 0.35, 0.45])
        done = False
        
        # メモリが空か確認
        self.assertEqual(len(self.agent.memory), 0)
        
        # データを追加
        self.agent.remember(state, action, reward, next_state, done)
        
        # メモリにデータが追加されたか確認
        self.assertEqual(len(self.agent.memory), 1)
    
    def test_memory_size_limit(self):
        """リプレイバッファのサイズ制限テスト"""
        state = np.array([0.1, 0.2, 0.3, 0.4])
        next_state = np.array([0.15, 0.25, 0.35, 0.45])
        
        # memory_size より多くデータを追加
        for i in range(3000):
            self.agent.remember(state, 0, 1.0, next_state, False)
        
        # メモリサイズが制限以下か確認
        self.assertLessEqual(len(self.agent.memory), self.agent.memory_size)
    
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
    
    def test_update_target_network(self):
        """ターゲットネットワーク更新テスト"""
        # モデルのパラメータを変更
        self.agent.model.set_weights(
            [w * 2 for w in self.agent.model.get_weights()]
        )
        
        # ターゲットネットワークを更新
        self.agent.update_target_network()
        
        # ターゲットネットワークが更新されたか確認
        model_weights = self.agent.model.get_weights()
        target_weights = self.agent.target_model.get_weights()
        
        for mw, tw in zip(model_weights, target_weights):
            np.testing.assert_array_almost_equal(mw, tw)
    
    def test_save_and_load_model(self):
        """モデルの保存と読み込みテスト"""
        import tempfile
        import os
        
        # 一時ディレクトリを作成
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model')
            
            # モデルを保存
            self.agent.save_model(model_path)
            
            # 新しいエージェントを作成
            agent2 = DQNAgent(
                state_size=4,
                action_size=2,
                learning_rate=0.001,
                discount_factor=0.99
            )
            
            # モデルを読み込み
            agent2.load_model(model_path)
            
            # モデルが読み込まれたか確認
            self.assertIsNotNone(agent2.model)
    
    def test_one_episode_training(self):
        """1エピソード訓練テスト"""
        state = self.env.reset()
        
        episode_reward = 0
        
        for step in range(100):
            action = self.agent.select_action(state, training=True)
            next_state, reward, done, _ = self.env.step(action)
            
            self.agent.remember(state, action, reward, next_state, done)
            
            # バッチサイズ分のデータがあれば訓練
            if len(self.agent.memory) >= self.agent.batch_size:
                self.agent.train_on_batch()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # エピソード報酬が正の値か確認
        self.assertGreater(episode_reward, 0)


if __name__ == '__main__':
    unittest.main()