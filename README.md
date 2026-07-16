# 🤖🐍 AIスネークゲーム (Django)

サーバーサイドの **AIがヘビを自動操作** するスネークゲームです。
ゲームのロジックとAIの思考は **Django（Python）** で動き、ブラウザは状態を描画するだけの薄いクライアントです。

## アーキテクチャ

```
ブラウザ (Canvas描画)  ──POST /api/step/──▶  Django
        ▲                                      │
        └──────── JSON（盤面の状態） ◀──────────┘
```

- **Djangoがゲームの真実の源（authoritative）**。盤面・ヘビ・エサ・スコアはすべてサーバーが保持します。
- ブラウザは一定間隔で `/api/step/` を呼び、サーバーのAIが決めた次の一手を反映して描画するだけです。
- ヘビの操作は完全にAI任せ（人間は操作しません）。

### 主要ファイル

| ファイル | 役割 |
|----------|------|
| `game/engine.py` | ゲームエンジン（状態と移動ルール） |
| `game/ai.py` | 探索AI本体＋戦略ディスパッチャ |
| `game/rl_agent.py` | 学習済みPPOモデルのアダプタ（gym_snakeで学習した重みを読み込む） |
| `game/store.py` | 進行中ゲームのインメモリ保管 |
| `game/views.py` | HTTP APIエンドポイント |
| `game/templates/game/index.html` | フロントエンド（描画とAIループ） |

## 3種類のAI

画面上部のセレクタで、ヘビを動かすAIを切り替えられます（ゲーム中でも切替可）。

| モード | 実装 | 盤面 |
|--------|------|------|
| 探索AI (BFS) | `game/ai.py` | 20×20 |
| 学習済みAI (features/MLP) | `game/rl_agent.py` + `examples/ppo_snake_features.zip` | 20×20 |
| 学習済みAI (grid/CNN) | `game/rl_agent.py` + `examples/ppo_snake_grid.zip` | **10×10 専用** |

### 1. 探索AI (BFS) — `game/ai.py`

`game/ai.py` は次の優先順で一手を決めます。

1. **エサへ最短で安全に向かう** — BFSでエサまでの最短経路を求め、食べた後も「自分のしっぽに到達できる」場合のみ実行（自分を閉じ込めない classic なテクニック）。
2. **生存を優先** — 安全な経路が無ければ、フラッドフィルで最も広い空間が残る方向へ逃げて時間を稼ぐ。
3. **最終手段** — どれも危険なら合法手を1つ選び、ゲームオーバーの判定はエンジンに委ねる。

この戦略により、20×20の盤面でヘビは長さ50〜150以上まで自滅せずに成長します。

### 2・3. 学習済みAI (PPO) — `game/rl_agent.py`

`gym_snake` 環境で **強化学習（PPO）させた重み** をそのまま読み込んで動かします。
`game/rl_agent.py` が、Djangoのゲーム状態を学習時と同一の観測に変換して推論し、
モデルの相対行動（直進/右折/左折）を絶対方向へ戻します。

- **features/MLP** … 11次元の特徴ベクトル観測。**盤面サイズに依存しない**ため、10×10で
  学習したモデルが20×20でもそのまま動きます。
- **grid/CNN** … `(3, H, W)` の画像観測 + カスタムCNN（`gym_snake.policies.SmallGridCNN`）。
  CNNのflatten→全結合層が学習時の **10×10 に固定**されているため、このAIを選ぶと盤面は
  自動的に10×10で作成されます（他のAIはサイズ非依存なので、その10×10盤面でも動作します）。
  20×20盤面でCNNを選ぶと、10×10で自動的に新規ゲームを開始します。
- **依存の無い環境でも安全**: `stable-baselines3`/`torch`（CNNは加えて `gym_snake`）が未インストール、
  またはモデルファイルが無い場合は自動的に探索AIへフォールバックし、フロント側では該当オプションが
  選択不可になります。盤面サイズが合わない場合もサーバー側で探索AIにフォールバックします。
- 初回推論の遅延を隠すため、ゲーム作成時に選択中のモデルをバックグラウンドで事前ロードします。

学習済みAIを使うには追加依存が必要です（学習方法は [`gym_snake/README.md`](gym_snake/README.md) 参照）:

```bash
pip install -e ".[train]"   # stable-baselines3 + torch（gym_snake も同時にインストール）
```

## セットアップと起動

```bash
pip install -r requirements.txt
python manage.py runserver
```

ブラウザで <http://127.0.0.1:8000/> を開き、「スタート」を押すとAIがプレイを始めます。

- **一時停止 / 再開**、**新規ゲーム**、**速度スライダー** で観戦を調整できます。
- ハイスコアはブラウザに自動保存されます。

## API

| メソッド・パス | 説明 |
|----------------|------|
| `GET /` | ゲーム画面 |
| `POST /api/new/` | 新規ゲームを作成（body: `{"strategy": ...}` で盤面サイズを決定）。`game_id`・初期状態・`strategies`（各AIの `available` と必要盤面サイズ `grid`）を返す |
| `POST /api/step/` | AIが一手進め、更新後の状態を返す（body: `{"game_id": "...", "strategy": "search"\|"rl"\|"rl_cnn"}`）。レスポンスの `strategy` は実際に使われたAI（フォールバック時は `"search"`） |
| `GET /api/state/?game_id=...` | 現在の状態を取得（進めない） |

## テスト

```bash
python manage.py test
```

エンジン・AI・APIの14テストが含まれています。

## 補足

- ルートの `index.html` は、**人間が矢印キーで操作する** スタンドアロン版（Django不要）です。AI版とは別物として残しています。
- ゲーム状態はプロセス内メモリに保持します。複数ワーカーで運用する場合は `game/store.py` をRedisやDBなどの共有バックエンドに差し替えてください。

---

## 🤖 強化学習用の環境 (gym_snake)

エージェントを **学習** させるための Gymnasium 形式の環境も同梱しています
（`gym_snake/` パッケージ）。行動空間・観測空間・報酬設計や使い方は
[`gym_snake/README.md`](gym_snake/README.md) を参照してください。

```bash
pip install -e .                         # gym_snake をインストール
python examples/random_agent.py --render # 動作確認
python examples/train_sb3.py --obs features --timesteps 200000  # PPOで学習
```

## このリポジトリの構成

| ディレクトリ / ファイル | 内容 |
|-------------------------|------|
| `index.html` | 人間が遊ぶスタンドアロン版（Django不要） |
| `snakeai/`, `game/`, `manage.py` | **AIが自動操作するDjango版**（サーバー側でAIが思考） |
| `gym_snake/` | **強化学習用のGymnasium環境**（エージェント学習用） |
| `examples/` | gym_snake のサンプル（ランダム方策・PPO学習） |
