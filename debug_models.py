"""
保存されたモデルの内容をデバッグするスクリプト
"""

import pickle
import os

print("\n" + "="*70)
print("🔍 Debugging Saved Models")
print("="*70)

# Q-Learning モデル
q_learning_path = 'data/models/qlearningagent_model.pkl'
if os.path.exists(q_learning_path):
    print(f"\n📌 Q-Learning Model: {q_learning_path}")
    with open(q_learning_path, 'rb') as f:
        q_obj = pickle.load(f)
    print(f"   Type: {type(q_obj)}")
    if isinstance(q_obj, dict):
        print(f"   Keys: {q_obj.keys()}")
        for key in q_obj.keys():
            print(f"      - {key}: {type(q_obj[key])}")
    else:
        print(f"   Attributes: {dir(q_obj)}")

# DQN モデル
dqn_path = 'data/models/dqnagent_model.pkl'
if os.path.exists(dqn_path):
    print(f"\n📌 DQN Model: {dqn_path}")
    try:
        with open(dqn_path, 'rb') as f:
            dqn_obj = pickle.load(f)
        print(f"   Type: {type(dqn_obj)}")
        if isinstance(dqn_obj, dict):
            print(f"   Keys: {dqn_obj.keys()}")
            for key in dqn_obj.keys():
                print(f"      - {key}: {type(dqn_obj[key])}")
        else:
            print(f"   Attributes: {dir(dqn_obj)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*70 + "\n")