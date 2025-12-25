import numpy as np


class ProductTagPairing:
    
    def __init__(self, horizontal_distance_factor=None, max_pairing_distance=None):
        if horizontal_distance_factor is None:
            horizontal_distance_factor = 1.0
        if max_pairing_distance is None:
            max_pairing_distance = 300
        
        self.horizontal_distance_factor = horizontal_distance_factor
        self.max_pairing_distance = max_pairing_distance
    
    def pair_products_and_tags(self, cropped_images):
        
        print(f"\n商品とタグのペアリングを開始...")
        
        # 商品とタグに分類
        products = [item for item in cropped_images if item.get('class') == 'product' and not item.get('filtered', False)]
        tags = [item for item in cropped_images if item.get('class') == 'tag' and not item.get('filtered', False)]
        
        print(f"商品: {len(products)}個, タグ: {len(tags)}個")
        
        pairs = []
        unpaired_products = []
        
        for product in products:
            product_box = product['box']
            # 商品の下底の中心座標
            product_bottom_center_x = (product_box[0] + product_box[2]) / 2
            product_bottom_y = product_box[3]  # 下底のy座標
            
            best_tag = None
            min_distance = float('inf')
            
            for tag in tags:
                tag_box = tag['box']
                # タグの上底の中心座標
                tag_top_center_x = (tag_box[0] + tag_box[2]) / 2
                tag_top_y = tag_box[1]  # 上底のy座標
                
                # 水平方向の距離
                horizontal_distance = abs(tag_top_center_x - product_bottom_center_x)
                
                # 垂直方向の距離
                vertical_distance = abs(tag_top_y - product_bottom_y)
                
                # ユークリッド距離を計算
                distance = np.sqrt(horizontal_distance**2 + vertical_distance**2)
                
                # 水平方向の許容範囲をチェック
                product_width = product_box[2] - product_box[0]
                if horizontal_distance > product_width * self.horizontal_distance_factor:
                    continue
                
                # 最大距離制限をチェック
                if distance > self.max_pairing_distance:
                    continue
                
                if distance < min_distance:
                    min_distance = distance
                    best_tag = tag
            
            if best_tag:
                pairs.append({
                    'product': product,
                    'tag': best_tag,
                    'distance': min_distance
                })
                print(f"  ペア作成: 商品#{product['index']} ↔ タグ#{best_tag['index']} (距離: {min_distance:.1f})")
            else:
                unpaired_products.append(product)
                print(f"  商品#{product['index']}: 対応するタグが見つかりませんでした")
        
        # 未ペアタグの計算（複数の商品とペアになるタグもカウント）
        paired_tag_indices = {pair['tag']['index'] for pair in pairs}
        unpaired_tags = [tag for tag in tags if tag['index'] not in paired_tag_indices]
        
        print(f"\nペアリング結果:")
        print(f"  ペア数: {len(pairs)}")
        print(f"  未ペア商品: {len(unpaired_products)}")
        print(f"  未ペアタグ: {len(unpaired_tags)}")
        
        # 重複ペアのチェックと表示
        tag_usage_count = {}
        for pair in pairs:
            tag_idx = pair['tag']['index']
            tag_usage_count[tag_idx] = tag_usage_count.get(tag_idx, 0) + 1
        
        duplicate_tags = {idx: count for idx, count in tag_usage_count.items() if count > 1}
        if duplicate_tags:
            print(f"\n情報: 以下のタグが複数の商品とペアになっています:")
            for tag_idx, count in duplicate_tags.items():
                print(f"  タグ#{tag_idx}: {count}個の商品とペア")
        
        return {
            'pairs': pairs,
            'unpaired_products': unpaired_products,
            'unpaired_tags': unpaired_tags
        }
