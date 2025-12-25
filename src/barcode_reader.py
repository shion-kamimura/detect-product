import cv2
from pyzbar import pyzbar
import easyocr
import re


class BarcodeReader:
    
    def __init__(self, product_registry):
        self.product_registry = product_registry
        # EasyOCR reader の初期化（数字のみ）
        print("OCRリーダーを読み込み中...")
        self.ocr_reader = easyocr.Reader(['en'])
        print("OCRリーダーの読み込みが完了")
    
    def detect_barcode_from_image(self, image_path):
        
        image = cv2.imread(image_path)
        
        if image is None:
            return []
        
        # 通常の画像でバーコード検出
        barcodes = pyzbar.decode(image)
        
        results = []
        for barcode in barcodes:
            # JAN-13 (EAN13) のみを対象とする
            if barcode.type == 'EAN13':
                barcode_data = barcode.data.decode('utf-8')
                results.append(barcode_data)
                print(f"    JAN-13検出: {barcode_data}")
            else:
                print(f"    スキップ ({barcode.type}): {barcode.data.decode('utf-8')}")
        
        # バーコードが見つからない場合はOCRで数字を読み取る
        if len(results) == 0:
            print(f"    バーコード検出失敗、OCRで数字を読み取り中...")
            ocr_result = self._read_numbers_with_ocr(image)
            if ocr_result:
                results.append(ocr_result)
        
        return results
    
    def _read_numbers_with_ocr(self, image):
        
        # OCRで文字認識
        ocr_results = self.ocr_reader.readtext(image)
        
        # 各テキストから13桁の数字のかたまりを探す
        for (bbox, text, prob) in ocr_results:
            # 数字のみを抽出
            numbers = re.sub(r'\D', '', text)
            
            if len(numbers) == 13:
                # ちょうど13桁の数字のかたまりを見つけた
                print(f"      OCR検出: '{text}' → JANコード: '{numbers}' (信頼度: {prob:.2f})")
                return numbers
            elif numbers:
                print(f"      OCR検出: '{text}' → 数字: '{numbers}' ({len(numbers)}桁, スキップ)")
        
        # 13桁の数字のかたまりが見つからなかった
        print(f"      13桁の数字のかたまりが検出できませんでした")
        return None
    
    def verify_product_by_barcode(self, tag_image_path, product_name):
        
        # タグからJANコードを検出
        barcodes = self.detect_barcode_from_image(tag_image_path)
        
        if not barcodes:
            print(f"    JANコードが検出できませんでした")
            return False, None
        
        detected_barcode = barcodes[0]  # 最初のJANコードを使用
        print(f"    検出されたJANコード: {detected_barcode}")
        
        # 商品辞書から期待されるJANコードを取得
        if product_name not in self.product_registry:
            print(f"    警告: '{product_name}'は登録されていません")
            return False, detected_barcode
        
        expected_barcode = self.product_registry[product_name].get('barcode')
        
        if not expected_barcode:
            print(f"    警告: '{product_name}'にJANコードが登録されていません")
            return None, detected_barcode
        
        # JANコードを比較
        is_match = detected_barcode == expected_barcode
        
        if is_match:
            print(f"    ✓ JANコード一致: {detected_barcode}")
        else:
            print(f"    ✗ JANコード不一致: 期待値={expected_barcode}, 検出値={detected_barcode}")
        
        return is_match, detected_barcode
