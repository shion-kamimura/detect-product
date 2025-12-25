import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

class SigLIPClassifier:
    
    def __init__(self):

        self.device = "mps"
        self.model_id = "models/siglip-base-patch16-224"
        
        
        print("SigLIPモデルを読み込み中...")
        
        # SigLIPの読み込み
        self.processor = AutoProcessor.from_pretrained(self.model_id, use_fast=True)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        
        print("SigLIPモデルの読み込みが完了\n")
        
        # 事前にテキスト特徴量を計算
        self.precompute_text_features()
    
    def precompute_text_features(self):
        
        texts = [
            "a photo of a product package, even if a small tag is attached",
            "a photo of a tag label"
        ]
        
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding="max_length"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            # 正規化
            self.text_features = outputs / outputs.norm(dim=-1, keepdim=True)
    
    def classify_image(self, image_path, return_probs=False):
        
        # 画像の読み込みと前処理
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        
        # デバイスに移動
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 画像特徴量の抽出
        with torch.no_grad():

            outputs = self.model.get_image_features(**inputs)

            # 正規化
            image_features = outputs / outputs.norm(dim=-1, keepdim=True)
            
            # コサイン類似度を計算
            similarities = (image_features @ self.text_features.T).squeeze(0)
            
            logit_scale = self.model.logit_scale.exp()
            logit_bias = self.model.logit_bias

            logits = (similarities * logit_scale) + logit_bias
            
            # 各クラスへの所属確率
            sigmoid_scores = torch.sigmoid(logits)
            
            # 2クラス分類のため、相対的な確率に正規化
            # ソフトマックスで合計が1になるように調整
            probs = torch.softmax(logits, dim=0)
            
            # 最も類似度が高いラベルを選択
            best_idx = logits.argmax().item()
        
        # インデックスをクラスに変換
        class_name = "product" if best_idx == 0 else "tag"
        
        if return_probs:
            prob_dict = {
                "product": probs[0].item(),
                "tag": probs[1].item(),
                "product_score": sigmoid_scores[0].item(),
                "tag_score": sigmoid_scores[1].item()
            }
            return class_name, prob_dict
        else:
            return class_name
    
    
    def match_product_images(self, image_path1, image_path2, return_similarity=False):
        
        # 画像の読み込み
        image1 = Image.open(image_path1).convert("RGB")
        image2 = Image.open(image_path2).convert("RGB")
        
        # 画像を個別に処理
        inputs1 = self.processor(images=image1, return_tensors="pt")
        inputs2 = self.processor(images=image2, return_tensors="pt")
        
        # デバイスに移動
        inputs1 = {k: v.to(self.device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(self.device) for k, v in inputs2.items()}
        
        # 画像特徴量を抽出
        with torch.no_grad():
            outputs1 = self.model.get_image_features(**inputs1)
            outputs2 = self.model.get_image_features(**inputs2)
            
            # 正規化
            features1 = outputs1 / outputs1.norm(dim=-1, keepdim=True)
            features2 = outputs2 / outputs2.norm(dim=-1, keepdim=True)
            
            # コサイン類似度を計算
            similarity = (features1 @ features2.T).squeeze().item()
        
        # 閾値で判定
        threshold = 0.7
        is_match = similarity >= threshold
        
        if return_similarity:
            return is_match, similarity
        else:
            return is_match
