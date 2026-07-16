# PPO 学習結果 — gym_snake

`gym_snake/Snake-v0` を Stable-Baselines3 の PPO で学習させた結果です。
2種類の観測（`features` / `grid`）でそれぞれ学習しています。

- **`features` (MLP)** … このページ下部。11次元の特徴ベクトル観測。
- **`grid` (CNN)** … [grid観測 + CNN の結果](#grid観測--cnn-の結果) を参照。

---

## `features` 観測 (MLP) の結果

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
- `obs_type="grid"` + `CnnPolicy` で空間構造を直接学習させる（下記参照）。
- 報酬シェイピングを切って（`reward_shaping=False`）純粋報酬で学習し、汎化を確認する。
- 盤面を大きくして（`grid_size=20`）難易度を上げる。

---

## `grid` 観測 + CNN の結果

生の `(3, 10, 10)` 画像観測（チャンネル `[体, 頭, エサ]`）を、小盤面向けの
カスタムCNN（[`gym_snake/policies.py`](../gym_snake/policies.py) の `SmallGridCNN`）で学習させた結果です。

> **なぜ標準の `CnnPolicy` ではダメか**: SB3標準の `NatureCNN` は 8×8 ストライド4 など
> Atari（〜84×84）向けの大きなカーネルを使うため、10×10 盤面では畳み込みで解像度が消えて
> しまいます。そこで stride=1・padding=1 で盤面解像度を保つ小型CNNを用意しました。

### 設定

| 項目 | 値 |
|------|------|
| アルゴリズム | PPO (`CnnPolicy` + `SmallGridCNN`) |
| 盤面 / 観測 | 10 × 10 / `grid`（`3×10×10` 画像） |
| 総ステップ | 3,000,000（8並列, CPU） |
| ハイパラ | `n_steps=1024`, `batch_size=2048`, `n_epochs=5`, `ent_coef=0.02`, `learning_rate=3e-4→0（線形減衰）` |
| 学習時間 | 約 49 分（CPU） |

### 学習曲線

![CNN learning curve](training_curve_cnn.png)

### ハイパラチューニングの知見（プラトー突破）

生ピクセルからの学習は、特徴ベクトル（`features`）より**サンプル効率が大きく劣ります**。
最初の設定（`ent_coef=0.01`・定数LR・`n_epochs=10`）では**平均スコア約2で頭打ち**に
なりました（局所解）。以下の変更でプラトーを突破しました。

- **探索を強化**（`ent_coef` 0.01 → 0.02）
- **学習率を線形減衰**（3e-4 → 0）で終盤の収束を安定化
- **ロールアウトを大きく・更新頻度を下げる**（`n_steps` 512→1024, `batch` 512→2048, `n_epochs` 10→5）で過学習を抑制

| ステップ | 平均スコア | 最大スコア |
|---------:|----------:|----------:|
| 0 (ランダム) | 0.07 | 1 |
| 500,000 | 0.60 | 2 |
| 1,000,000 | 4.17 | 9 |
| 1,500,000 | 7.53 | 25 |
| 2,000,000 | 11.70 | 27 |
| 2,500,000 | 13.73 | 32 |
| 2,750,000 | **14.57** | 28 |
| 3,000,000 | 14.37 | 29 |

- ランダム0点 → **平均約14.5点・最大32点**（別シード30エピソード評価では平均約18点）。
- `features`のMLP（平均約23点）には及びませんが、これは想定通りです。MLPは危険・エサ方向が
  観測に直接エンコードされているのに対し、CNNは**生の画像から同等の情報を自力で抽出**する
  必要があるためです。曲線を見ると3Mでもまだ上昇傾向で、さらにステップを増やせば伸びます。

### 学習済みCNNモデルでのプレイ例（スコア28）

```
+----------+
|......oo..|
|....ooooo.|
|...oo..oo.|
|..oo..oo@.|
|..o.....o.|
|..oo....o.|
|...oo..oo.|
|....oooo..|
|*....oo...|
|..........|
+----------+   @=頭 o=体 *=エサ
```

### 再現・利用方法

```bash
pip install -e ".[train]"
python examples/train_sb3.py --obs grid --grid 10 --timesteps 3000000
```

```python
import gymnasium as gym, gym_snake
from gym_snake.policies import SmallGridCNN   # モデル読み込みに必要
from stable_baselines3 import PPO

model = PPO.load("examples/ppo_snake_grid.zip", device="cpu")
env = gym.make("gym_snake/Snake-v0", grid_size=10, obs_type="grid", render_mode="ansi")
obs, _ = env.reset(seed=0)
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    done = terminated or truncated
print("score:", info["score"])
```

> 学習済みモデル `ppo_snake_grid.zip`（約20MB）は CNN の重みとオプティマイザ状態を含みます。
