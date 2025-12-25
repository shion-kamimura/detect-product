import torch
from PIL import Image
import os
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class ObjectDetector:
    
    def __init__(self):
        
        self.device = "mps"
        self.model_id = "models/grounding-dino-base"
        
        # Grounding DINO の読み込み
        print("物体検出モデルを読み込み中...")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        print("物体検出モデルの読み込みが完了")
    
    def detect_objects(self, image_path, text_prompt, threshold=None):
        
        if threshold is None:
            threshold = 0.18
        
        # 画像の読み込み
        image = Image.open(image_path).convert("RGB")

        max_size = 2304
        if max(image.size) > max_size:
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            print(f"画像をリサイズ: {image.size}")
        
        # 入力の準備
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)
        
        # 推論
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 結果の後処理
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            target_sizes=[image.size[::-1]]
        )[0]
        
        return {
            "image": image,
            "boxes": results["boxes"].cpu().numpy(),
            "scores": results["scores"].cpu().numpy(),
            "labels": results["labels"]
        }
    
    def crop_detected_objects(self, results, output_dir="output/cropped1", 
                             max_objects=None, max_width_ratio=None, 
                             max_height_ratio=None, padding_ratio=None):
        
        if max_width_ratio is None:
            max_width_ratio = 0.8
        if max_height_ratio is None:
            max_height_ratio = 0.8
        if padding_ratio is None:
            padding_ratio = 0.1
        
        print(f"\n検出されたオブジェクトを切り出し中...")
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        image = results["image"]
        image_width, image_height = image.size
        
        cropped_images = []
        
        # インデックスのリストを作成
        indices = list(range(len(results["boxes"])))
        
        # max_objectsが指定されている場合は制限
        if max_objects:
            indices = indices[:max_objects]
            print(f"上位{max_objects}個のオブジェクトのみ処理")
        
        filtered_count = 0
        saved_count = 0
        for rank, idx in enumerate(indices, 1):
            box = results["boxes"][idx]
            score = results["scores"][idx]
            label = results["labels"][idx]
            x1, y1, x2, y2 = map(int, box)
            
            # バウンディングボックスのサイズをチェック
            box_width = x2 - x1
            box_height = y2 - y1
            width_ratio = box_width / image_width
            height_ratio = box_height / image_height
            
            # バウンディングボックスで切り出し
            cropped = image.crop((x1, y1, x2, y2))
            
            # 大きすぎるかチェック
            is_filtered = width_ratio > max_width_ratio or height_ratio > max_height_ratio
            
            # Grounding DINOのラベルから分類を決定
            detected_class = None
            if "product" in label.lower():
                detected_class = "product"
            elif "tag" in label.lower():
                detected_class = "tag"
            
            # ファイル名を生成
            prefix = "filtered_" if is_filtered else ""
            filename = f"{prefix}object_{saved_count+1:03d}_{label}_{score:.2f}.png"
            filepath = os.path.join(output_dir, filename)
            
            # 保存（除外される場合もデバッグ用に保存）
            cropped.save(filepath)
            saved_count += 1
            
            if is_filtered:
                filtered_count += 1
                reason = []
                if width_ratio > max_width_ratio:
                    reason.append(f"幅比 {width_ratio:.2%} > {max_width_ratio:.2%}")
                if height_ratio > max_height_ratio:
                    reason.append(f"高さ比 {height_ratio:.2%} > {max_height_ratio:.2%}")
                print(f"  [除外] オブジェクト{rank}: {', '.join(reason)} → {filename} ")
                cropped_images.append({
                    "index": len(cropped_images) + 1,
                    "original_index": int(idx),
                    "filepath": filepath,
                    "label": label,
                    "class": detected_class,
                    "score": float(score),
                    "box": box,
                    "width_ratio": width_ratio,
                    "height_ratio": height_ratio,
                    "filtered": True,
                    "image": cropped
                })
            else:
                cropped_images.append({
                    "index": len(cropped_images) + 1,
                    "original_index": int(idx),
                    "filepath": filepath,
                    "label": label,
                    "class": detected_class,
                    "score": float(score),
                    "box": box,
                    "width_ratio": width_ratio,
                    "height_ratio": height_ratio,
                    "filtered": False,
                    "image": cropped
                })
        
        if filtered_count > 0:
            print(f"{filtered_count}個のオブジェクトを除外")
        print(f"{len(cropped_images)}個のオブジェクトを保存: {output_dir}")
        print(f"  検索対象: {len([x for x in cropped_images if not x.get('filtered', False)])}個")
        print(f"  除外: {len([x for x in cropped_images if x.get('filtered', False)])}個")
        
        return cropped_images
