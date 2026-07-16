# PPO 学習結果 — gym_snake

`gym_snake/Snake-v0` を Stable-Baselines3 の PPO で学習させた結果です。
3種類の観測（`features` / `grid` / `ego`）でそれぞれ学習しています。

- **`features` (MLP)** … このページ下部。11次元の特徴ベクトル観測。
- **`grid` (CNN)** … [grid観測 + CNN の結果](#grid観測--cnn-の結果) を参照（`ego` に置き換え済み）。
- **`ego` (CNN・サイズ非依存)** … [ego観測 + CNN の結果](#ego観測--cnn-の結果サイズ非依存) を参照。**現在の推奨CNN**。

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

> **注**: このモデルはflatten層が10×10盤面に固定されるため、サイズ非依存の
> [`ego` 観測モデル](#ego観測--cnn-の結果サイズ非依存)（性能も大幅に上）に置き換えられました。
> モデルファイル `ppo_snake_grid.zip` はリポジトリから削除済みです（`train_sb3.py --obs grid` で再現可能）。
> 以下は当時の記録として残しています。

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

---

## `ego` 観測 + CNN の結果（サイズ非依存）

**現在の推奨CNN構成です。** 観測を「盤面の生画像」から「ヘビ自身から見た自己中心
（egocentric）ビュー」に変えることで、**盤面サイズ非依存**と**大幅な性能向上**を
同時に達成しました（[`gym_snake/obs.py`](../gym_snake/obs.py)）。

### 観測の設計 — `(5, 11, 11)` 固定

| ch | 内容 |
|----|------|
| 0 | 局所ビュー: 危険セル（壁 + 体。しっぽは次ティックに空くため除外） |
| 1 | 局所ビュー: エサ |
| 2 | ミニマップ: 体全体 |
| 3 | ミニマップ: エサ（ピーク正規化） |
| 4 | ミニマップ: 頭（ピーク正規化） |

- **局所ビュー**は頭を中心とした11×11の切り出しで、**進行方向が常に上**になるよう回転。
  「正面の危険」が常に同じピクセルに現れるため、相対行動（直進/右折/左折）と完全に整合します。
- **ミニマップ**は盤面全体を11×11に縮約（面積平均）したもので、局所ビュー外のエサも見えます。
- 観測形状が盤面サイズによらず固定なので、**flatten型CNN（`SmallGridCNN`）がそのまま使え**、
  1つの重みが10×10でも20×20でも動きます（リロードや再バインド不要）。

### なぜ生画像＋グローバルプーリングではダメだったか

先に「CoordConv + adaptive pooling で全結合層をサイズ非依存化する」アプローチ
（`AnyGridCNN`）を試しましたが、**1Mステップで平均0.03点と学習が完全に停止**しました。
プーリングで「頭と首の相対位置（＝進行方向）」「頭のすぐ隣の危険」という1ピクセル精度の
相対幾何情報が潰れることが原因です。ego観測は回転と中心化によってこの情報を
**固定ピクセル位置に構造的に埋め込む**ため、この問題が起きません。

### カリキュラム学習（10×10 → 20×20）

設定は grid 版と同じチューニング済みレシピ（`n_steps=1024, batch=2048, n_epochs=5,
ent_coef=0.02, lr=3e-4→0 線形減衰`、8並列）。

![ego learning curve](training_curve_ego.png)

**フェーズ1: 10×10 で 3M ステップ（約51分）**

| ステップ | 平均スコア | 最大 |
|---------:|----------:|-----:|
| 250,000 | 13.93 | 22 |
| 750,000 | 27.30 | 39 |
| 1,500,000 | 35.70 | 51 |
| 2,500,000 | **45.17** | 67 |

わずか250kステップで旧grid版CNNの最終性能（14.5）に到達し（**約11倍のサンプル効率**）、
最終的に features MLP（約23）の2倍の平均45.2点に達しました。

**ゼロショット転移**: 10×10で学習したモデルをそのまま20×20で評価 → **平均53.5点・最大99**。
ファインチューニング前から高性能に転移します。

**フェーズ2: 20×20 で 1.5M ステップのファインチューニング（約29分）**

| ステップ | 平均スコア（20×20） | 最大 |
|---------:|----------:|-----:|
| 250,000 | **61.43** | 93 |
| 1,000,000 | 53.13 | 111 |
| 1,500,000 | 53.40 | 98 |

ベストは **平均61.4点**。ファインチューニング後も10×10で平均33.8点を維持しており、
**1つのモデルが両サイズで実用的に動作**します。

### 3方式の比較（10×10、gym環境）

| 方式 | 平均スコア | 備考 |
|------|----------:|------|
| ランダム | 0.07 | |
| grid CNN（旧・サイズ固定） | 14.5 | 3Mステップ |
| features MLP | 23.0 | 2Mステップ |
| **ego CNN（サイズ非依存）** | **45.2** | 3Mステップ |

### 再現・利用方法

```bash
pip install -e ".[train]"
# 単一サイズで学習する場合
python examples/train_sb3.py --obs ego --grid 10 --timesteps 3000000
```

```python
import gymnasium as gym, gym_snake
from stable_baselines3 import PPO

model = PPO.load("examples/ppo_snake_ego.zip", device="cpu")
# 観測形状が固定なので、どの盤面サイズでも同じモデルがそのまま動く
for grid in (10, 20):
    env = gym.make("gym_snake/Snake-v0", grid_size=grid, obs_type="ego")
    obs, _ = env.reset(seed=0)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    print(grid, "->", info["score"])
```
