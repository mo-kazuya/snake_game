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
| **報酬** | エサ `+1.0` ／ 死亡 `-1.0` ／ 毎ステップ `-0.005` ／ （任意）エサに近づくと `±0.05` のシェイピング |
| **終了 terminated** | 壁 or 自分の体に衝突（または盤面を埋め尽くしてクリア） |
| **打ち切り truncated** | エサを取らずに `max_steps_without_food`（既定 `H*W`）ステップ経過 |

### パラメータ

| 引数 | 既定 | 説明 |
|------|------|------|
| `grid_size` | `12` | 盤面の一辺のマス数（`>=5`） |
| `obs_type` | `"grid"` | `"grid"` または `"features"` |
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

```bash
# grid観測 + CNN で学習
python examples/train_sb3.py --obs grid --grid 10 --timesteps 3000000
```

## テスト

```bash
pytest gym_snake/tests/
```

Gymnasium の `check_env` によるAPI準拠チェックを両観測モードで含みます。
