# PPO 学習結果 — gym_snake

`gym_snake/Snake-v0` を Stable-Baselines3 の PPO で学習させた結果です。

## 設定

| 項目 | 値 |
|------|------|
| アルゴリズム | PPO (`MlpPolicy`) |
| 盤面 | 10 × 10 |
| 観測 | `features`（11次元ベクトル） |
| 並列環境数 | 8 |
| 総ステップ数 | 2,000,000 |
| 主なハイパラ | `n_steps=512`, `batch_size=512`, `gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0.01`, `lr=3e-4` |
| 学習時間 | 約 300 秒（CPU, 8並列） |

## 学習曲線

![learning curve](training_curve.png)

`deterministic=True` で各チェックポイント30エピソードを評価した平均スコアです。
（スコア = 食べたエサの数）

| ステップ | 平均スコア | 最大スコア | 平均リターン |
|---------:|----------:|----------:|-----------:|
| 0 (ランダム) | 0.07 | 1 | -1.11 |
| 250,000 | 20.17 | 36 | 24.34 |
| 500,000 | 20.23 | 33 | 24.37 |
| 750,000 | 24.03 | 47 | 29.06 |
| 1,000,000 | 19.53 | 28 | 23.63 |
| 1,250,000 | 21.87 | 35 | 26.27 |
| 1,500,000 | 24.40 | 44 | 29.48 |
| 1,750,000 | 20.40 | 34 | 24.49 |
| 2,000,000 | 22.97 | 39 | 27.81 |

- **ランダム方策のスコアは 0** → 25万ステップで平均20点に急上昇し、以降は平均20〜24点で安定。
- 10×10（最大スコア97）で **平均約23点・最大47点**。エージェントは壁と自分を避けつつエサを追い、盤面をとぐろ状に埋める挙動を学習しました。

## 学習済みモデルでのプレイ例（スコア36）

```
+----------+
|..........|
|oooo......|
|o..o......|
|o..o......|
|o..o.*....|
|o..o......|
|o..oooooo.|
|ooooooo@o.|
|ooo....oo.|
|..oooooo..|
+----------+   @=頭 o=体 *=エサ
```

## 再現方法

```bash
pip install -e ".[train]"
python examples/train_sb3.py --obs features --grid 10 --timesteps 2000000
```

学習済みモデル `ppo_snake_features.zip` の読み込みと評価:

```python
import gymnasium as gym, gym_snake
from stable_baselines3 import PPO

model = PPO.load("examples/ppo_snake_features.zip")
env = gym.make("gym_snake/Snake-v0", grid_size=10, obs_type="features", render_mode="ansi")
obs, _ = env.reset(seed=0)
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated
    print(env.render())
print("score:", info["score"])
```

## さらに伸ばすには

- ステップ数を増やす（5M〜10M）と、より長く盤面を埋められるようになります。
- `obs_type="grid"` + `CnnPolicy` で空間構造を直接学習させる。
- 報酬シェイピングを切って（`reward_shaping=False`）純粋報酬で学習し、汎化を確認する。
- 盤面を大きくして（`grid_size=20`）難易度を上げる。
