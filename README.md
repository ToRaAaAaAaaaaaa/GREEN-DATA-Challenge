# SMBC Group GREEN×DATA Challenge 2025 for Recruiting
# スペイン電力価格予測プロジェクト

## プロジェクト概要
スペインの電力市場における実際の価格(price_actual)を予測する機械学習モデルの構築

## データセット
- **学習データ**: train.csv
- **テストデータ**: test.csv
- **対象地域**: Valencia, Madrid, Bilbao, Barcelona, Seville

## 実施内容

### 1. データ前処理
- **欠損値補完**: KNNImputerを使用した数値列の補完(k=5)
- **カテゴリカル変数**: One-Hotエンコーディングによる数値化
- **正規化**: StandardScalerによるスケーリング

### 2. 特徴量エンジニアリング

#### 時間的特徴量
- 年(year)
- 月(month)
- 曜日(day_of_week)
- 時間(hour)
- 祝日フラグ(holidays) - スペインの祝日データを使用

#### 気象データの次元削減
- 5都市の気象データ(温度、気圧、湿度、風速など12特徴量)をPCAで2次元に圧縮
- PC1, PC2として新規特徴量を作成

### 3. モデル構築・評価

#### 使用モデル
1. **LightGBM**
2. **XGBoost**
3. **Random Forest**
4. **LSTM** (系列長24時間)
5. **Ridge回帰**

#### 評価指標
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score

### 4. 特徴量重要度分析
LightGBMを使用して特徴量の重要度を算出・可視化

### 5. ハイパーパラメータチューニング
GridSearchCVによるXGBoostの最適化
- n_estimators: 4000
- max_depth: 5
- learning_rate: 0.001
- その他正則化パラメータ調整

### 6. 最終予測
最適化されたXGBoostモデルでテストデータを予測し、`predictions.csv`として出力

## 主要な特徴量(重要度上位)
1. PC1, PC2 (気象データの主成分)
2. holidays (祝日フラグ)
3. generation_fossil_hard_coal (石炭発電量)
4. total_load_actual (総負荷実績)
5. barcelona_pressure (バルセロナ気圧)
6. generation_nuclear (原子力発電量)
7. その他発電量・気象関連特徴量

## 技術スタック
- **データ処理**: pandas, numpy
- **可視化**: matplotlib, seaborn, japanize-matplotlib
- **機械学習**: scikit-learn, LightGBM, XGBoost, TensorFlow/Keras
- **時系列分析**: Prophet, ARIMA/SARIMAX
- **その他**: holidays(祝日データ)

## 実行環境
- Google Colab (GPU: T4)
- Python 3.10.18

## ファイル構成
```
├── green_data (2).ipynb  # メインノートブック
├── data/
│   ├── train.csv
│   └── test.csv
└── predictions.csv       # 予測結果出力
```

## 実行方法
1. Google Driveをマウント
2. 必要なライブラリをインストール
3. ノートブックを順番に実行
4. predictions.csvが生成される

## 注意点
- 時系列データのため、train/test分割は時系列を考慮(後半20%をテスト)
- TimeSeriesSplitによるクロスバリデーション実施
- EarlyStoppingを使用した過学習防止
