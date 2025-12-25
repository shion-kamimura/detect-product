import os
import json
import warnings
from object_detector import ObjectDetector
from barcode_reader import BarcodeReader
from pairing import ProductTagPairing
from visualizer import Visualizer
from classifier import SigLIPClassifier

# 警告を非表示にする
warnings.filterwarnings('ignore')

class DrugstoreDetector:
    
    def __init__(self):
        # 各コンポーネントの初期化
        self.object_detector = ObjectDetector()
        self.pairing = ProductTagPairing()
        self.visualizer = Visualizer()
        self.siglip_classifier = SigLIPClassifier()
        
        # 商品辞書の初期化
        self.product_registry = {}
        
        # BarcodeReaderを初期化
        self.barcode_reader = BarcodeReader(self.product_registry)
    
    def register_product(self, product_name, reference_image_path, barcode=None):
        
        self.product_registry[product_name] = {
            'image_path': reference_image_path,
            'barcode': barcode
        }
        print(f"商品を登録: {product_name} -> {reference_image_path}" + (f" (バーコード: {barcode})" if barcode else ""))
    
    def process_all_objects(self, cropped_images, target_product_name=None):
        
        print(f"\n画像内容を分析中...")
        
        # 除外されていないオブジェクトのみ処理
        active_images = [item for item in cropped_images if not item.get('filtered', False)]
        filtered_images = [item for item in cropped_images if item.get('filtered', False)]
        
        print(f"処理対象: {len(active_images)}個 (除外: {len(filtered_images)}個)")
        
        # Grounding DINOのラベルから分類結果を表示
        unclassified_count = 0
        for item in active_images:
            if item['class'] is None:
                unclassified_count += 1
            print(f"[{item['index']}/{len(cropped_images)}] ラベル: {item['label']} → 分類: {item['class']}")
        
        # Grounding DINOで分類できなかったオブジェクトをSigLIPで分類
        if unclassified_count > 0:
            print(f"\n{unclassified_count}個の未分類オブジェクトをSigLIPで分類中...")
            for item in active_images:
                if item['class'] is None:
                    print(f"[{item['index']}] SigLIPで分類中: {item['label']}")
                    classified, probs = self.siglip_classifier.classify_image(item['filepath'], return_probs=True)
                    # productと判定された場合のみクラスを付与
                    if classified == 'product':
                        item['class'] = 'product'
                        print(f"  → product")
                        print(f"     確率: 商品={probs['product']:.1%}, タグ={probs['tag']:.1%}")
                    else:
                        print(f"  → 未分類のまま (tag判定)")
                        print(f"     確率: 商品={probs['product']:.1%}, タグ={probs['tag']:.1%}")
        else:
            print(f"\nすべてのオブジェクトがGrounding DINOで分類されました")
        
        # 商品とタグをペアリング
        pairing_result = self.pairing.pair_products_and_tags(cropped_images)
        
        # 分類結果でフィルタリング
        product_images = [item for item in active_images if item['class'] == 'product']
        tag_images = [item for item in active_images if item['class'] == 'tag']
        
        print(f"\n最終分類結果: product={len(product_images)}個, tag={len(tag_images)}個")
        
        results = []
        
        # 除外されたオブジェクトを結果に追加
        for item in filtered_images:
            results.append({
                "index": item['index'],
                "class": item.get('class', 'filtered'),
                "label": item['label'],
                "width_ratio": item.get('width_ratio'),
                "height_ratio": item.get('height_ratio'),
                "matched": False,
                "paired_with": None,
                "barcode_verified": None,
                "barcode_data": None
            })
        
        # ペアリング情報を含めてタグを追加
        for item in tag_images:
            paired_product_index = None
            for pair in pairing_result['pairs']:
                if pair['tag']['index'] == item['index']:
                    paired_product_index = pair['product']['index']
                    break
            
            results.append({
                "index": item['index'],
                "class": "tag",
                "label": item['label'],
                "matched": False,
                "paired_with": paired_product_index,
                "barcode_verified": None,
                "barcode_data": None
            })
        
        # 特定商品の検索が指定されている場合、SigLIPで商品マッチング
        matched_products = []
        if target_product_name:
            print(f"\nproductクラス({len(product_images)}個)から '{target_product_name}' を検索中...")
            
            # 辞書から参照画像を取得
            if target_product_name not in self.product_registry:
                print(f"    警告: '{target_product_name}'は登録されていません")
            else:
                reference_image_path = self.product_registry[target_product_name]['image_path']
                
                # SigLIPで商品マッチング
                print(f"  SigLIPで商品マッチングを実行...")
                for item in product_images:
                    print(f"[{item['index']}] 判定中: {item['label']}")
                    try:
                        is_match, similarity = self.siglip_classifier.match_product_images(
                            reference_image_path,
                            item['filepath'],
                            return_similarity=True
                        )
                        if is_match:
                            print(f"    ✓ 一致 (類似度: {similarity:.3f})")
                            matched_products.append(item)
                        else:
                            print(f"    ✗ 不一致 (類似度: {similarity:.3f})")
                    except Exception as e:
                        print(f"    エラー: {e}")
                
                print(f"\n検索結果: {len(matched_products)}個の一致する商品が見つかりました")
                
                # 一致した商品のタグからバーコードを検証
                if matched_products:
                    print(f"\n一致した商品のバーコード検証を開始...")
                    for item in matched_products:
                        # ペアになっているタグを探す
                        paired_tag = None
                        for pair in pairing_result['pairs']:
                            if pair['product']['index'] == item['index']:
                                paired_tag = pair['tag']
                                break
                        
                        if paired_tag:
                            print(f"\n商品#{item['index']}のペアタグ#{paired_tag['index']}を検証中...")
                            verified, barcode_data = self.barcode_reader.verify_product_by_barcode(
                                paired_tag['filepath'],
                                target_product_name
                            )
                            
                            # 検証結果を保存
                            item['barcode_verified'] = verified
                            item['barcode_data'] = barcode_data
                            
                            # タグの情報も更新
                            for tag_result in results:
                                if tag_result['index'] == paired_tag['index']:
                                    tag_result['barcode_verified'] = verified
                                    tag_result['barcode_data'] = barcode_data
                                    break
                        else:
                            print(f"\n商品#{item['index']}: ペアのタグが見つかりませんでした")
                            item['barcode_verified'] = None
                            item['barcode_data'] = None
        
        # 結果を整形（ペアリング情報とバーコード検証結果を含む）
        for item in product_images:
            is_matched = item in matched_products if target_product_name else False
            
            # ペアになっているタグのインデックスを取得
            paired_tag_index = None
            for pair in pairing_result['pairs']:
                if pair['product']['index'] == item['index']:
                    paired_tag_index = pair['tag']['index']
                    break
            
            results.append({
                "index": item['index'],
                "class": "product",
                "label": item['label'],
                "matched": is_matched,
                "paired_with": paired_tag_index,
                "barcode_verified": item.get('barcode_verified'),
                "barcode_data": item.get('barcode_data')
            })
        
        print(f"\n全{len(results)}個のオブジェクトの処理が完了(product: {len(product_images)}個, tag: {len(tag_images)}個)")
        
        return results, matched_products, pairing_result
    
    def save_results_to_json(self, results, output_path="output/results1.json"):
        """結果をJSONファイルに保存"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n結果をJSONファイルに保存: {output_path}")


def main():
    
    # 検出器の初期化
    detector = DrugstoreDetector()
    
    # 商品の登録（商品名、参照画像、バーコード番号）
    # detector.register_product("アレグラFX28錠", "input/reference/allegra_fx_28.jpg", "230606349269")
    detector.register_product("AGアレルカットc15ml", "input/reference/ag_allercut_c_15.jpeg", "4987107673756")
    # detector.register_product("シエロヘアカラーEXクリーム4Aアッシュブラウン", "input/reference/ciero_haircolor_ex_4a_ashbrown.jpg", "4987205284731")
    # detector.register_product("新ビオフェルミンs350錠", "input/reference/biofermin_s_350.jpeg", "4987306054783")
    # detector.register_product("エスセレクトキズあて滅菌パッドS12枚入り", "input/reference/s_select_kizuate_sterile_pad_s_12.jpeg", "4904820142505")
    # detector.register_product("エスセレクト和紙ばんそうこう10mm", "input/reference/s_select_washi_bandage_10mm.jpeg", "4566322160052")
    # detector.register_product("ペリペラティント05", "input/reference/peripera_tint_05.jpeg", "4573198753370")
    
    # 画像パスの設定
    image_path = "input/drugstore1.jpeg"
    
    # 検出したい物体のプロンプト
    text_prompt = "a product. a tag."
    
    # 処理するオブジェクトの最大数
    max_objects = None 
    
    # バウンディングボックスのサイズ上限（画像に対する比率）
    max_width_ratio = 0.8 
    max_height_ratio = 0.8
    
    # 検索する商品名
    target_product_name = "AGアレルカットc15ml" 
    
    # 物体検出の実行
    print(f"\n画像を解析中: {image_path}")
    detection_results = detector.object_detector.detect_objects(
        image_path=image_path,
        text_prompt=text_prompt,
        threshold=0.18
    )
    
    # 結果のサマリーを表示
    detector.visualizer.print_detection_summary(detection_results)
    
    # 結果を可視化して保存
    output_path = "output/result_specific1.jpeg"
    detector.visualizer.visualize_results(detection_results, save_path=output_path, show=False)
    
    # 検出されたオブジェクトを個別に保存
    cropped_images = detector.object_detector.crop_detected_objects(
        detection_results, 
        max_objects=max_objects,
        max_width_ratio=max_width_ratio,
        max_height_ratio=max_height_ratio
    )
    
    processed_results, matched_products, pairing_result = detector.process_all_objects(
        cropped_images, 
        target_product_name=target_product_name
    )
    
    # 結果のサマリーを表示
    detector.visualizer.print_summary(processed_results)
    
    # 一致した商品のバウンディングボックスを可視化（バーコード一致のみ）
    matched_output_path = None
    if target_product_name and matched_products:
        # バーコードが一致した商品のみをフィルタリング
        barcode_verified_products = [
            item for item in matched_products 
            if item.get('barcode_verified') == True
        ]
        
        if barcode_verified_products:
            matched_output_path = "output/result_matched1.jpeg"
            detector.visualizer.visualize_matched_products(
                detection_results, 
                barcode_verified_products, 
                save_path=matched_output_path, 
                show=False
            )
            print(f"\nバーコード一致した商品: {len(barcode_verified_products)}個")
        else:
            print(f"\nバーコードが一致した商品はありませんでした")
    
    # 結果をJSONファイルに保存
    detector.save_results_to_json(processed_results)
    
    print(f"\n処理完了")
    print(f"  全検出結果画像: {output_path}")
    if matched_output_path:
        print(f"  一致商品画像: {matched_output_path}")
    print(f"  ペアリング結果:")
    print(f"    ペア数: {len(pairing_result['pairs'])}")
    print(f"    未ペア商品: {len(pairing_result['unpaired_products'])}")
    print(f"    未ペアタグ: {len(pairing_result['unpaired_tags'])}")
    
    return detector, detection_results, matched_products, pairing_result


if __name__ == "__main__":
    detector, detection_results, matched_products, pairing_result = main()