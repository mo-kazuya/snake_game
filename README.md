# 🤖🐍 AIスネークゲーム (Django)

サーバーサイドの **AIがヘビを自動操作** するスネークゲームです。
ゲームのロジックとAIの思考は **Django（Python）** で動き、ブラウザは状態を描画するだけの薄いクライアントです。

## アーキテクチャ

```
ブラウザ (Canvas描画)  ──WebSocket ws/game/<id>/──▶  Django (Channels)
        ▲                                              │
        └───────────── JSON（盤面の状態） ◀────────────┘
```

- **Djangoがゲームの真実の源（authoritative）**。盤面・ヘビ・エサ・スコアはすべてサーバーが保持します。
- ゲーム開始時に `POST /api/new/` でゲームを作成した後、ブラウザは1本のWebSocket
  （`ws/game/<game_id>/`）を張りっぱなしにし、一定間隔でメッセージを送って
  サーバーのAIが決めた次の一手を反映して描画するだけです（毎ティックHTTPリクエストは
  発生しません）。
- ヘビの操作は完全にAI任せ（人間は操作しません）。

### 主要ファイル

| ファイル | 役割 |
|----------|------|
| `game/engine.py` | ゲームエンジン（状態と移動ルール） |
| `game/ai.py` | 探索AI本体＋戦略ディスパッチャ |
| `game/rl_agent.py` | 学習済みPPOモデルのアダプタ（gym_snakeで学習した重みを読み込む） |
| `game/store.py` | 進行中ゲームのインメモリ保管 |
| `game/views.py` | HTTP APIエンドポイント（ゲーム作成・状態取得） |
| `game/consumers.py` | WebSocketコンシューマー（毎ティックのAI着手） |
| `game/routing.py` | WebSocket URLルーティング |
| `snakeai/asgi.py` | ASGIエントリポイント（HTTP/WebSocketの振り分け） |
| `game/templates/game/index.html` | フロントエンド（描画とAIループ） |

## 4種類のAI

画面上部のセレクタで、ヘビを動かすAIを切り替えられます（ゲーム中でも切替可）。

| モード | 実装 | 盤面 |
|--------|------|------|
| 探索AI (BFS) | `game/ai.py` | 任意（既定 20×20） |
| 学習済みAI (features/MLP) | `game/rl_agent.py` + `examples/ppo_snake_features.zip` | 任意（既定 20×20） |
| 学習済みAI (ego/CNN) | `game/rl_agent.py` + `examples/ppo_snake_ego.zip` | 任意（既定 20×20） |
| 学習済みAI (ego/Transformer) | `game/rl_agent.py` + `examples/ppo_snake_transformer.zip` | 任意（既定 20×20） |

### 1. 探索AI (BFS) — `game/ai.py`

`game/ai.py` は次の優先順で一手を決めます。

1. **エサへ最短で安全に向かう** — BFSでエサまでの最短経路を求め、食べた後も「自分のしっぽに到達できる」場合のみ実行（自分を閉じ込めない classic なテクニック）。
2. **生存を優先** — 安全な経路が無ければ、フラッドフィルで最も広い空間が残る方向へ逃げて時間を稼ぐ。
3. **最終手段** — どれも危険なら合法手を1つ選び、ゲームオーバーの判定はエンジンに委ねる。

この戦略により、20×20の盤面でヘビは長さ50〜150以上まで自滅せずに成長します。

### 2〜4. 学習済みAI (PPO) — `game/rl_agent.py`

`gym_snake` 環境で **強化学習（PPO）させた重み** をそのまま読み込んで動かします。
`game/rl_agent.py` が、Djangoのゲーム状態を学習時と同一の観測に変換して推論し、
モデルの相対行動（直進/右折/左折）を絶対方向へ戻します。

- **features/MLP** … 11次元の特徴ベクトル観測。**盤面サイズに依存しない**ため、10×10で
  学習したモデルが20×20でもそのまま動きます。
- **ego/CNN** … **自己中心（egocentric）観測** + CNN。頭を中心に進行方向が常に上になるよう
  回転した11×11の局所ウィンドウと、盤面全体を11×11に縮約したミニマップの計5チャンネル
  （`gym_snake/obs.py`）。観測形状が `(5, 11, 11)` 固定なので **こちらも盤面サイズ非依存**です。
  10×10で3M→20×20で1.5Mのカリキュラム学習後、**8〜40の複数サイズ混合で計5M
  ステップの追加ファインチューニング済み**。8×8〜40×40の全サイズで実用的な強さです
  （20×20で平均約59エサ、30×30で約71エサ、40×40で約75エサ）。
- **ego/Transformer** … 同じego観測を **ViT型Transformer**（セル単位パッチ+CLSトークン、
  `gym_snake.policies.EgoTransformer`、d_model=128・4層）で処理するモデル。
  **探索AIを教師とする模倣学習（BC）で事前学習し、PPOでファインチューニング**する
  2段階レシピ（`examples/train_transformer_v2.py`）で学習した現在の**リポジトリ最強モデル**
  です（同一プロトコル比較で全6サイズがego/CNNを上回り、平均67.5エサ。20×20で約73、
  40×40で約95）。PPO単独で学習したv1はCNNに大きく劣っており（平均8.9）、
  「Transformerの弱い帰納バイアスを教師データで補う」効果の実証になっています。
  詳細な分析は `examples/TRAINING_RESULTS.md` を参照。
- **依存の無い環境でも安全**: `stable-baselines3`/`torch`（CNN/Transformerは加えて `gym_snake`）が未インストール、
  またはモデルファイルが無い場合は自動的に探索AIへフォールバックし、フロント側では該当オプションが
  選択不可になります。
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

`channels`/`daphne` がインストール済みだと `manage.py runserver` は自動的にDaphne
（ASGIサーバー）で起動し、同じポートでHTTPとWebSocketの両方を捌きます。追加の
コマンドやポート設定は不要です。

ブラウザで <http://127.0.0.1:8000/> を開き、「スタート」を押すとAIがプレイを始めます。

- **盤面サイズ**（10×10 / 14×14 / 20×20 / 30×30 / 40×40）をセレクタで選べます。
  ゲーム中に変更すると新しいサイズで新規ゲームが始まります。3種のAIはすべて
  サイズ非依存なので、どのサイズでも動作します（APIでは8〜50の任意サイズを指定可）。
- **一時停止 / 再開**、**新規ゲーム**、**速度スライダー** で観戦を調整できます。
- ハイスコアは**盤面サイズごとに**ブラウザへ自動保存されます。

## API

| メソッド・パス | 説明 |
|----------------|------|
| `GET /` | ゲーム画面 |
| `POST /api/new/` | 新規ゲームを作成（body: `{"strategy": ..., "grid": 8〜50}`。省略時は20。モデルが特定サイズを要求する場合はそちらを優先）。`game_id`・初期状態・`strategies`（各AIの `available` と必要盤面サイズ `grid`）を返す |
| `GET /api/state/?game_id=...` | 現在の状態を取得（進めない） |
| `WS /ws/game/<game_id>/` | 毎ティックのAI着手用WebSocket。クライアントが `{"strategy": "search"\|"rl"\|"rl_cnn"\|"rl_trf"}` を送るたびにサーバーが一手進め、`{"game_id", "direction", "strategy", "event", "state"}` を1メッセージ返す（`strategy` は実際に使われたAI。フォールバック時は `"search"`）。`game_id` が存在しない場合は `{"error": "unknown game_id"}` を送ってから接続を閉じる（close code `4404`） |

以前の `POST /api/step/`（毎ティックHTTPリクエスト）はこのWebSocketに置き換えられ、廃止されました。

## テスト

```bash
python manage.py test
```

エンジン・AI・HTTP API・WebSocketコンシューマーのテストが含まれています。

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
