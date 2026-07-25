# gym_snake 🐍 — Gymnasium environment for training Snake agents

A self-contained [Gymnasium](https://gymnasium.farama.org/) environment for the
game of Snake, designed for reinforcement-learning experiments.

## インストール

```bash
pip install -e .            # コア (gymnasium + numpy)
pip install -e ".[train]"   # + stable-baselines3 と torch (学習用)
pip install -e ".[render]"  # + pygame (human描画用)
```

## 使い方

```python
import gymnasium as gym
import gym_snake  # 登録のために import が必要

env = gym.make("gym_snake/Snake-v0", grid_size=12, obs_type="grid")
obs, info = env.reset(seed=0)

done = False
while not done:
    action = env.action_space.sample()      # 0=直進, 1=右折, 2=左折
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

print(info)  # {'score': ..., 'length': ..., 'steps': ...}
```

## 環境仕様

| 項目 | 内容 |
|------|------|
| **行動空間** | `Discrete(3)` — `0`=直進 / `1`=右折 / `2`=左折（相対方向。逆走による即死が無く学習しやすい） |
| **観測 `obs_type="grid"`**（既定） | `Box(0,1, shape=(3, H, W), float32)`。チャンネル `[体, 頭, エサ]`。CNN向け |
| **観測 `obs_type="features"`** | `Box(0,1, shape=(11,), float32)`。危険センサ3＋進行方向one-hot4＋エサ方向4。MLP向け・高速 |
| **観測 `obs_type="ego"`** | `Box(0,1, shape=(5, W, W), float32)`（既定 `W=11`、`ego_window` で変更可）。頭中心・進行方向が上になるよう回転した局所ビュー＋盤面ミニマップ（`gym_snake/obs.py`）。**形状が盤面サイズ非依存**なので1つのモデルが任意の盤面で動く。CNN向け・**推奨** |
| **報酬** | エサ `+1.0` ／ 死亡 `-1.0` ／ 毎ステップ `-0.005` ／ （任意）エサに近づくと `±0.05` のシェイピング |
| **終了 terminated** | 壁 or 自分の体に衝突（または盤面を埋め尽くしてクリア） |
| **打ち切り truncated** | エサを取らずに `max_steps_without_food`（既定 `H*W`）ステップ経過 |

### パラメータ

| 引数 | 既定 | 説明 |
|------|------|------|
| `grid_size` | `12` | 盤面の一辺のマス数（`>=5`） |
| `obs_type` | `"grid"` | `"grid"` / `"features"` / `"ego"` |
| `ego_window` | `None` (=11) | `obs_type="ego"` の観測窓の一辺（**奇数・5以上**）。大きくすると頭の周囲をより広く見え、ミニマップの縮約も粗くなくなる。学習時と推論時で必ず同じ値にすること |
| `reward_shaping` | `True` | エサへの接近/離反に応じた小報酬の有無 |
| `max_steps_without_food` | `H*W` | 空回り防止の打ち切り上限 |
| `render_mode` | `None` | `"ansi"` / `"rgb_array"` / `"human"` |

## サンプル

```bash
# ランダム方策で動作確認（ansi描画付き）
python examples/random_agent.py --episodes 3 --grid 10 --render

# PPOで学習（stable-baselines3 が必要）
python examples/train_sb3.py --obs features --timesteps 200000
python examples/train_sb3.py --obs grid --timesteps 500000
```

> 参考: 観測（`features`）だけを見る単純な貪欲方策でも 10×10 盤面で平均スコア
> 約19点に達します（ランダムは0点）。観測に十分な学習シグナルが含まれています。

### 学習結果

PPOで実際に学習させた結果（学習曲線・スコア推移・学習済みモデル）は
[`examples/TRAINING_RESULTS.md`](../examples/TRAINING_RESULTS.md) にまとめています。

- **`features` 観測 (MLP)**: 2Mステップで平均スコア約23点（ランダムは0点）。
- **`grid` 観測 (CNN)**: 3Mステップで平均スコア約14.5点。小盤面向けのカスタムCNN
  （[`gym_snake/policies.py`](policies.py) の `SmallGridCNN`）を使用。生ピクセルからの
  学習はサンプル効率が劣るため、探索強化・学習率減衰でプラトーを突破しています。
- **`ego` 観測 (CNN・サイズ非依存)**: 10×10で3Mステップ → **平均約45点**（gridの3倍、
  featuresの2倍）。観測形状が固定のため同じモデルが任意の盤面で動き、20×20への
  ゼロショット転移で平均53.5点、1.5Mステップのファインチューニングで**平均61.4点**。
  さらに**複数サイズ（8〜40）を同一VecEnvに混在させた計5Mステップの混合ファインチューニング**で
  大盤面の弱点を解消（30×30: 3.8→71.3点、40×40: 1.6→74.9点。40×40が最強サイズに）。
  配分と性能プロファイルのトレードオフ分析は `examples/TRAINING_RESULTS.md` を参照。
- **`ego` 観測 (Transformer)**: 特徴抽出器をViT型 `EgoTransformer` に差し替えた構成。
  PPO単独の公平比較（v1、`--arch transformer`）ではサンプル効率がCNNの約1/3〜1/7で
  明確に劣位でしたが、**探索AIのデモで模倣学習してからPPOする2段階レシピ**
  （v2、`examples/train_transformer_v2.py`）では**全盤面サイズでCNNを上回り**
  6サイズ平均67.5点（CNNは52.7点）に達しました。

```bash
# ego観測 + CNN で学習（推奨）
python examples/train_sb3.py --obs ego --grid 10 --timesteps 3000000
```

## 2匹対戦環境 (`SnakeBattle-v0`)

単独学習では相手のいる盤面を経験できないため、**2匹が同じ盤面・同じエサを取り合う
対戦環境** `gym_snake/SnakeBattle-v0`（[`gym_snake/envs/battle_env.py`](envs/battle_env.py)）
を同梱しています。ルールは Django サーバーと同じ権威的エンジン（`game.engine.GameState`）
をそのまま使うので、**学習時と本番のルールが完全一致**します（エサ共有・相手の体や
正面衝突での死・片方が死んでももう片方は続行）。

- **エージェントは snake 0** を操作し、snake 1 は設定可能な**相手方策**が動かします。
- 観測は単独時と**同じ固定形状の ego 観測**ですが、`opponent_cells` に相手の体を
  重ねて渡すため、モデルは「動く相手」を危険チャンネル上で認識できます。観測・行動
  空間が単独版と同一なので、**単独学習済みTransformerからそのままウォームスタート**できます。
- 基本報酬は `SnakeEnv` と同一（価値ヘッドの再利用のため）。任意で対戦特化のボーナス
  （相手撃破 `reward_opp_death`／勝敗 `reward_win`・`reward_lose`）を上乗せできます。

```python
import gymnasium as gym
import gym_snake  # 登録のため

env = gym.make("gym_snake/SnakeBattle-v0", grid_size=12, opponent="search",
               reward_win=1.0, reward_lose=-1.0)
obs, info = env.reset(seed=0)   # obs.shape == (5, 11, 11)
```

`SnakeBattleEnv` も `ego_window` を受け取ります（単独版と同じ観測窓を指定すれば、
広視野モデルからもそのままウォームスタートできます）。

相手方策 `opponent` は次の文字列で指定します（`make_opponent`）:

| spec | 相手の動き |
|------|-----------|
| `"search"`（既定） | BFS/フラッドフィルの探索AI（本番と同じ強敵） |
| `"random"` | 一様ランダムな**安全手**（カリキュラムの多様性用） |
| `"model:<path>"` | 凍結したPPOチェックポイント（**自己対戦**用。遅延ロードでpicklable） |

この環境で**単独Transformerをファインチューニングして対戦特化モデルを作る**
2段階レシピが [`examples/train_transformer_battle.py`](../examples/train_transformer_battle.py)
です（ウォームスタート→任意の対戦BC→対戦PPO、勝率ゲート付き）。生成物
`ppo_snake_transformer_battle.zip` は Django 側で `rl_trf_battle` として選択できます。

```bash
# 既存Transformerからウォームスタートし、探索AI+ランダム相手にPPO
python examples/train_transformer_battle.py

# 自己対戦（前世代のチェックポイントを相手に）
python examples/train_transformer_battle.py \
    --opponents search,model:examples/ppo_snake_transformer_battle.zip
```

**LoRA での代替レシピ**: ベースを凍結し低ランクアダプタだけを学習してマージする
ドロップイン版が [`examples/train_transformer_lora.py`](../examples/train_transformer_lora.py)
です。ベースは明確に上回りますが、本モデル（約0.88M・注意機構を凍結）ではフルBCに一歩
及びません。詳細な比較は [`examples/TRAINING_RESULTS.md`](../examples/TRAINING_RESULTS.md) 参照。

## テスト

```bash
pytest gym_snake/tests/
```

Gymnasium の `check_env` によるAPI準拠チェックを全観測モードで含みます
（単独 `SnakeEnv`・対戦 `SnakeBattleEnv` の両方）。
