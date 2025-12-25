from PIL import ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import Counter


class Visualizer:
    
    def __init__(self):
        # フォントの設定
        self.font = ImageFont.load_default()
    
    def visualize_results(self, results, save_path=None, show=False):
        
        image = results["image"].copy()
        draw = ImageDraw.Draw(image)
        
        # 各検出結果を描画
        color = (255, 0, 0)
        
        for idx, (box, score, label) in enumerate(zip(
            results["boxes"], 
            results["scores"], 
            results["labels"]
        )):
            x1, y1, x2, y2 = box
            
            # バウンディングボックスを描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
            
            # ラベルとスコアを描画
            text = f"{idx+1}: {label} ({score:.2f})"
            
            # テキストの背景を描画
            text_bbox = draw.textbbox((x1, y1), text, font=self.font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1), text, fill="white", font=self.font)
        
        # 結果を表示
        if show:
            plt.figure(figsize=(15, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"検出された物体数: {len(results['boxes'])}")
            plt.tight_layout()
            plt.show()
        
        # 保存
        if save_path:
            image.save(save_path)
            print(f"検出結果を保存しました: {save_path}")
        
        return image
    
    def visualize_matched_products(self, results, matched_items, save_path=None, show=False):
        
        if not matched_items:
            print("一致した商品がないため、可視化をスキップ")
            return None
        
        image = results["image"].copy()
        draw = ImageDraw.Draw(image)
        
        # 一致した商品のバウンディングボックスを描画
        color = (0, 255, 0)
        
        for item in matched_items:
            box = item['box']
            x1, y1, x2, y2 = box
            
            # バウンディングボックスを描画
            draw.rectangle([x1, y1, x2, y2], outline=color, width=5)
            
            # ラベルを描画
            text = f"#{item['index']}: {item['label']} ({item['score']:.2f})"
            
            # テキストの背景を描画
            text_bbox = draw.textbbox((x1, y1), text, font=self.font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1), text, fill="white", font=self.font)
        
        # 結果を表示
        if show:
            plt.figure(figsize=(15, 10))
            plt.imshow(image)
            plt.axis('off')
            plt.title(f"一致した商品: {len(matched_items)}個")
            plt.tight_layout()
            plt.show()
        
        # 保存
        if save_path:
            image.save(save_path)
            print(f"一致した商品の検出結果を保存しました: {save_path}")
        
        return image
    
    def print_detection_summary(self, results):
        
        print(f"検出結果サマリー")
        print(f"検出された物体数: {len(results['boxes'])}")
        
        # ラベルごとの検出数を集計
        label_counts = Counter(results['labels'])
        
        print(f"\nラベルごとの検出数:")
        for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: {count}個")
    
    def print_summary(self, vlm_results):

        print(f"\n処理結果サマリー")
        
        product_count = sum(1 for r in vlm_results if r.get('class') == 'product')
        tag_count = sum(1 for r in vlm_results if r.get('class') == 'tag')
        matched_count = sum(1 for r in vlm_results if r.get('matched') == True)
        
        # バーコード検証結果の集計（Noneを除外）
        barcode_verified_count = sum(1 for r in vlm_results 
                                     if r.get('class') == 'product' 
                                     and r.get('matched') == True 
                                     and r.get('barcode_verified') == True)
        barcode_failed_count = sum(1 for r in vlm_results 
                                   if r.get('class') == 'product' 
                                   and r.get('matched') == True 
                                   and r.get('barcode_verified') == False)
        barcode_no_tag_count = sum(1 for r in vlm_results 
                                   if r.get('class') == 'product' 
                                   and r.get('matched') == True 
                                   and r.get('barcode_verified') is None)
        
        print(f"処理したオブジェクト数: {len(vlm_results)}個")
        print(f"  product: {product_count}個")
        print(f"  tag: {tag_count}個")
        if matched_count > 0:
            print(f"一致した商品: {matched_count}個")
            if barcode_verified_count > 0:
                print(f"  バーコード検証成功: {barcode_verified_count}個")
            if barcode_failed_count > 0:
                print(f"  バーコード検証失敗: {barcode_failed_count}個")
            if barcode_no_tag_count > 0:
                print(f"  バーコード未検証(タグなし): {barcode_no_tag_count}個")
