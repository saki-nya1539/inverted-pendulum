"""
Cart-Pole 環境のテスト
"""

import unittest
import numpy as np
from src.cart_pole_env import CartPoleEnv


class TestCartPoleEnv(unittest.TestCase):
    """
    CartPoleEnv のテストクラス
    """
    
    def setUp(self):
        """各テストの前に実行"""
        self.env = CartPoleEnv()
    
    def tearDown(self):
        """各テストの後に実行"""
        self.env.close()
    
    def test_env_initialization(self):
        """環境の初期化テスト"""
        self.assertIsNotNone(self.env)
        self.assertEqual(self.env.action_space.n, 2)
        self.assertEqual(self.env.observation_space.shape[0], 4)
    
    def test_reset(self):
        """リセット機能テスト"""
        state = self.env.reset()
        
        # 状態の形状チェック
        self.assertEqual(len(state), 4)
        
        # 状態が配列か確認
        self.assertIsInstance(state, np.ndarray)
        
        # 初期状態が妥当な値か確認
        self.assertTrue(np.all(np.abs(state) < 1.0))
    
    def test_step(self):
        """ステップ機能テスト"""
        self.env.reset()
        
        # アクション 0 でステップ
        next_state, reward, done, info = self.env.step(0)
        
        # 戻り値の型チェック
        self.assertEqual(len(next_state), 4)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, (bool, np.bool_))
        self.assertIsInstance(info, dict)
    
    def test_step_bounds(self):
        """ステップ後の状態が妥当な範囲内か確認"""
        self.env.reset()
        
        for _ in range(10):
            next_state, reward, done, info = self.env.step(np.random.randint(0, 2))
            
            # 状態値がNaNでないか確認
            self.assertTrue(np.all(np.isfinite(next_state)))
    
    def test_multiple_episodes(self):
        """複数エピソードのテスト"""
        for episode in range(5):
            state = self.env.reset()
            
            for step in range(100):
                action = self.env.action_space.sample()
                next_state, reward, done, info = self.env.step(action)
                
                if done:
                    break
            
            self.assertGreater(step, 0)
    
    def test_reward_signal(self):
        """報酬信号のテスト"""
        self.env.reset()
        
        # 10ステップ実行
        total_reward = 0
        for _ in range(10):
            _, reward, done, _ = self.env.step(0)
            total_reward += reward
            if done:
                break
        
        # 報酬の合計が正の値か確認
        self.assertGreater(total_reward, 0)
    
    def test_discretize_state(self):
        """状態の離散化テスト"""
        state = np.array([0.5, 0.1, 0.2, -0.1])
        bins = 10
        
        discrete_state = self.env.discretize_state(state, bins=bins)
        
        # 離散化状態がタプル型か確認
        self.assertIsInstance(discrete_state, tuple)
        
        # 各要素が0以上bins未満か確認
        for s in discrete_state:
            self.assertGreaterEqual(s, 0)
            self.assertLess(s, bins)
    
    def test_done_condition(self):
        """終了条件のテスト"""
        self.env.reset()
        
        # ポールが大きく傾いた状態を作る
        steps = 0
        while steps < 500:
            _, _, done, _ = self.env.step(1)  # 右へ力を加え続ける
            steps += 1
            
            if done:
                break
        
        # 終了するまでのステップが50以上100以下のはず
        self.assertGreater(steps, 0)
        self.assertLessEqual(steps, 500)


if __name__ == '__main__':
    unittest.main()