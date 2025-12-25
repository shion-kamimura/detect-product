# 商品検出システム（Product Detection System）

ドラッグストアや小売店舗における商品とタグの自動検出・マッチングシステムです。Grounding DINOとSigLIPを活用し、物体検出、商品識別、バーコード読み取り、商品-タグペアリングを実行します。

## 主な機能

- **物体検出**: Grounding DINOを使用した高精度な物体検出
- **分類**: 商品とタグの自動分類（Grounding DINO + SigLIP）
- **ペアリング**: 商品とタグの自動マッチング
- **商品検索**: SigLIPによる参照画像との類似度比較
- **バーコード検証**: EasyOCRとpyzbarによるバーコード読み取りと検証
- **可視化**: 検出結果のバウンディングボックス表示

## システム構成

```
src/
├── main.py              # メイン実行ファイル
├── object_detector.py   # Grounding DINOによる物体検出
├── classifier.py        # SigLIPによる商品分類・マッチング
├── barcode_reader.py    # バーコード読み取り
├── pairing.py          # 商品-タグペアリング
└── visualizer.py       # 結果の可視化
```


## インストール

1. リポジトリのクローン
```bash
git clone <repository-url>
cd detect-product
```

2. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

3. モデルの配置

以下のモデルを`models/`ディレクトリに配置してください：
- `grounding-dino-base/` - Grounding DINOモデル
- `siglip-base-patch16-224/` - SigLIPモデル

## 使い方

### 基本的な使用例

```python
from main import DrugstoreDetector

# 検出器の初期化
detector = DrugstoreDetector()

# 商品の登録（商品名、参照画像パス、バーコード番号）
detector.register_product(
    "商品名",
    "input/reference/product_image.jpg",
    "4987107673756"
)

# 物体検出の実行
detection_results = detector.object_detector.detect_objects(
    image_path="input/drugstore1.jpeg",
    text_prompt="a product. a tag.",
    threshold=0.18
)

# 検出されたオブジェクトを処理
cropped_images = detector.object_detector.crop_detected_objects(
    detection_results,
    max_width_ratio=0.8,
    max_height_ratio=0.8
)

# 商品検索とペアリング
results, matched_products, pairing_result = detector.process_all_objects(
    cropped_images,
    target_product_name="商品名"
)
```

### メインスクリプトの実行

```bash
cd src
python main.py
```

## 設定パラメータ

### 物体検出

- `threshold`: 検出の信頼度閾値（デフォルト: 0.18）
- `text_prompt`: 検出対象のテキストプロンプト（例: "a product. a tag."）
- `max_width_ratio`: バウンディングボックスの最大幅比率（デフォルト: 0.8）
- `max_height_ratio`: バウンディングボックスの最大高さ比率（デフォルト: 0.8）

### 商品マッチング

- `similarity_threshold`: SigLIPの類似度閾値（デフォルト: 0.85）

## 出力ファイル

処理結果は`output/`ディレクトリに保存されます：

- `results.json` - 検出結果のJSON
- `result_specific.jpeg` - 全検出結果の可視化
- `result_matched.jpeg` - 一致した商品のみの可視化
- `cropped/` - 切り出されたオブジェクト画像

### JSON出力形式

```json
[
  {
    "index": 1,
    "class": "product",
    "label": "a product",
    "matched": true,
    "paired_with": 2,
    "barcode_verified": true,
    "barcode_data": "4987107673756"
  }
]
```

## プロジェクト構造

```
detect-product/
├── README.md
├── requirements.txt
├── input/
│   ├── drugstore1.jpeg      # 入力画像
│   └── reference/           # 参照画像（商品マスター）
├── models/
│   ├── grounding-dino-base/
│   └── siglip-base-patch16-224/
├── output/
│   ├── results.json
│   ├── result_specific.jpeg
│   ├── result_matched.jpeg
│   └── cropped/             # 切り出し画像
└── src/
    ├── main.py
    ├── object_detector.py
    ├── classifier.py
    ├── barcode_reader.py
    ├── pairing.py
    └── visualizer.py
```

## 処理フロー

1. **物体検出** - Grounding DINOで商品とタグを検出
2. **分類** - Grounding DINOとSigLIPで商品/タグを分類
3. **ペアリング** - 商品とタグの位置関係から自動ペアリング
4. **商品検索** - SigLIPで参照画像との類似度比較
5. **バーコード検証** - ペアのタグからバーコードを読み取り検証
6. **結果出力** - JSON保存と可視化

## 技術スタック

- **物体検出**: Grounding DINO (Zero-shot Object Detection)
- **画像分類**: SigLIP (Vision-Language Model)
- **OCR**: EasyOCR
- **バーコード**: pyzbar
- **深層学習**: PyTorch, Transformers
- **画像処理**: Pillow, OpenCV

## トラブルシューティング

### モデル読み込みエラー

モデルファイルが正しく配置されているか確認してください：
```bash
ls models/grounding-dino-base/
ls models/siglip-base-patch16-224/
```

### メモリ不足

画像サイズが大きい場合、自動でリサイズされます（最大2304px）。それでもメモリ不足の場合は、画像を事前に縮小してください。

### バーコード読み取り失敗

- タグ画像の解像度が十分か確認
- バーコードが鮮明に写っているか確認
- バーコード番号が商品辞書に正しく登録されているか確認