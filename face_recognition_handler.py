import os
import pickle
import cv2
import face_recognition
from utils import preprocess_image
from config import ENCODINGS_FILE


class FaceRecognitionHandler:
    """人臉識別處理器"""
    
    def __init__(self):
        """初始化人臉識別處理器"""
        self.known_face_encodings = []
    
    def load_or_create_encodings(self, image_paths):
        """
        載入或創建人臉編碼
        
        Args:
            image_paths: 圖片路径列表
        
        Returns:
            人臉編碼列表
        """
        # 嘗試從檔案載入編碼
        if os.path.exists(ENCODINGS_FILE):
            try:
                with open(ENCODINGS_FILE, "rb") as f:
                    print("從文件加載預計算的臉部特徵...")
                    self.known_face_encodings = pickle.load(f)
                    return self.known_face_encodings
            except Exception as e:
                print(f"載入編碼檔案失敗: {e}")
                print("重新計算臉部特徵...")

        # 重新計算編碼
        print("計算臉部特徵中...")
        known_encodings = []
        valid_images = 0
        failed_images = []

        for i, path in enumerate(image_paths):
            print(f"處理進度: {i+1}/{len(image_paths)} - {os.path.basename(path)}")
            
            try:
                if not os.path.exists(path):
                    print(f"❌ 文件不存在: {path}")
                    failed_images.append(path)
                    continue
                
                # 預處理圖片
                image = preprocess_image(path)
                if image is None:
                    failed_images.append(path)
                    continue
                
                # 檢測人臉
                face_locations = face_recognition.face_locations(image, model="hog")
                
                if not face_locations:
                    print(f"❌ 未檢測到人臉")
                    failed_images.append(path)
                    continue

                # 選擇最大的人臉
                if len(face_locations) > 1:
                    largest_face = max(face_locations, 
                                     key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
                    face_locations = [largest_face]

                # 提取特徵編碼
                encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)
                
                if encodings:
                    known_encodings.append(encodings[0])
                    valid_images += 1
                    print(f"✅ 成功提取特徵")
                else:
                    print(f"❌ 特徵提取失敗")
                    failed_images.append(path)
                    
            except Exception as e:
                print(f"❌ 處理錯誤: {str(e)}")
                failed_images.append(path)

        # 結果統計
        self._print_processing_summary(len(image_paths), valid_images, failed_images)

        # 保存編碼
        if valid_images >= 3:
            self._save_encodings(known_encodings)
        else:
            print(f"\n❌ 錯誤: 只有 {valid_images} 張有效照片，至少需要3張")

        self.known_face_encodings = known_encodings
        return known_encodings
    
    def recognize_faces(self, frame):
        """
        識別畫面中的人臉
        
        Args:
            frame: 攝影機畫面
        
        Returns:
            (face_locations, face_encodings, face_distances)
        """
        # 轉換為 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 檢測人臉位置
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        
        if not face_locations:
            return [], [], []
        
        # 獲取人臉編碼
        face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
        
        # 計算距離
        face_distances_list = []
        for face_encoding in face_encodings:
            distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            face_distances_list.append(distances)
        
        return face_locations, face_encodings, face_distances_list
    
    def _print_processing_summary(self, total_images, valid_images, failed_images):
        """打印處理結果統計"""
        print(f"\n{'='*50}")
        print(f"處理完成統計:")
        print(f"總圖片數: {total_images}")
        print(f"成功處理: {valid_images}")
        print(f"失敗數量: {len(failed_images)}")
        print(f"成功率: {valid_images/total_images*100:.1f}%")
    
    def _save_encodings(self, encodings):
        """保存編碼到檔案"""
        try:
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(encodings, f)
            print(f"\n✅ 特徵編碼已保存到 {ENCODINGS_FILE}")
        except Exception as e:
            print(f"保存編碼檔案失敗: {e}")