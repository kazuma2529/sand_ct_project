# 粒子セグメンテーションと接触分析

X 線 CT 画像から粒子（砂粒）を一粒ずつ認識し、3D 構造を再構成して接触分析を行うプロジェクトです。

## 機能

1. **粒子のセグメンテーション**

   - U-Net ベースの深層学習モデルによる粒子の自動認識
   - 2D スライス画像から粒子のマスクを生成

2. **3D 構造の再構成**

   - 2D マスク画像を 3D スタックとして再構成
   - 粒子のラベル付けと個別識別

3. **接触分析**
   - 粒子間の接触判定
   - 接触面積の計算
   - 接触数の自動カウント

## 環境設定

1. 必要なライブラリのインストール:

```bash
pip install -r requirements.txt
```

## データセット構造

以下のようなディレクトリ構造を使用しています：

```
sand_ct_project/
├── train.py/
│   └── train.py          # 学習用スクリプト
├── predict.py/
│   └── predict.py        # 推論用スクリプト
├── model.py/
│   └── model.py          # U-Netモデル定義
├── dataset.py/
│   └── dataset.py        # データセット処理
├── analyze_3d.py         # 3D化と接触判定
├── edit_images.py        # 画像強調処理スクリプト
├── requirements.txt/
│   └── requirements.txt  # 依存パッケージ
└── data/
    ├── train/
    │   ├── images/           # 元の学習用画像
    │   ├── images_enhanced/  # コントラスト・シャドウ強調された学習用画像
    │   ├── annotations/      # labelmeで作成したポリゴンアノテーション（.json）
    │   └── masks/            # 学習用マスク（アノテーションから生成）
    ├── val/
    │   ├── images/      # 検証用画像
    │   └── masks/       # 検証用マスク
    └── test/
        ├── images/      # テスト用画像
        └── predicted_masks/  # 予測マスク出力先
```

### データ要件

- 画像はグレースケールの PNG ファイル
- マスクは 2 値画像（0 または 255）
- 画像とマスクは同じ解像度である必要があります

### データ処理手順

1. オリジナル CT 画像を`data/train/images/`に配置
2. `edit_images.py`を実行して画像の強調処理を行い`data/train/images_enhanced/`に保存
3. labelme を使用して画像にポリゴンアノテーションを作成し、結果の.json ファイルを`data/train/annotations/`に保存
4. アノテーションファイルからマスク画像を生成し、`data/train/masks/`に保存

## 使用方法

### 1. 画像の強調処理

```bash
python edit_images.py
```

コントラストとシャドウを強調した画像が`data/train/images_enhanced/`に保存されます。

### 2. 学習

```bash
python train.py
```

学習の進捗は以下のファイルに保存されます：

- `best_model.pth`: 最良のモデルの重み
- `learning_curves.png`: 学習曲線のグラフ
- `training.log`: 学習のログ

### 3. 推論

```bash
python predict.py
```

テスト画像に対する予測結果は `data/test/predicted_masks/` に保存されます。

### 4. 接触分析

```bash
python analyze_3d.py
```

分析結果は以下のファイルに保存されます：

- `particle_contacts.csv`: 各粒子の接触情報
  - 粒子 ID
  - 接触数
  - 総接触面積
  - 接触している粒子の ID と接触面積
- `analysis.log`: 分析のログ

## モデルについて

- U-Net アーキテクチャを使用
- 入力：グレースケール画像（1 チャンネル）
- 出力：2 値マスク画像（1 チャンネル）
- エンコーダー：5 段階のダウンサンプリング
- デコーダー：5 段階のアップサンプリング
- スキップコネクション：対応するエンコーダー層からデコーダー層へ

## ハイパーパラメータ

- バッチサイズ：4
- エポック数：50
- 学習率：0.001
- オプティマイザー：Adam
- 損失関数：Binary Cross Entropy
- 早期停止：10 エポック改善がない場合
- 学習率スケジューラー：検証損失が 5 エポック改善がない場合に 0.1 倍

## エラー処理とログ

- 各処理で詳細なログを記録
- データの検証とエラーチェック
- 例外処理による安定した実行
