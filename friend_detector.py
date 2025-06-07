import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import pickle
import os
import time
import threading
import pyautogui

# 設定參數
FRIEND_NAME = "Your Friend"
CONFIDENCE_THRESHOLD = 0.37
TRIGGER_DISTANCE = 180
ENCODINGS_FILE = "known_face_encodings.pkl"
VIDEO_PATH = "update.mp4"  # 確保這個檔案存在
NO_FRIEND_FRAMES_THRESHOLD = 15

# 中文文字繪製函數
def cv2_puttext_chinese(img, text, position, font_size, color):
    """在OpenCV圖像上繪製中文文字"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        font = ImageFont.truetype("msyh.ttc", font_size)  # Windows 微軟雅黑
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # 備用字體
        except:
            font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 影片播放器類
class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.is_playing = False
        self.window_name = "Video Player"
        self.thread = None
        
        # 檢查影片檔案是否存在
        if not os.path.exists(video_path):
            print(f"警告: 影片檔案不存在: {video_path}")
            self.video_exists = False
        else:
            self.video_exists = True
            self.cap = cv2.VideoCapture(video_path)
    
    def play(self):
        if not self.video_exists:
            print("無法播放影片：檔案不存在")
            return
            
        if not self.is_playing:
            self.is_playing = True
            self.thread = threading.Thread(target=self._play_loop)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        self.is_playing = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
    
    def _play_loop(self):
        if not self.video_exists:
            return
            
        try:
            while self.is_playing:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到開頭
                
                while self.is_playing:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    
                    # 獲取螢幕尺寸並調整影片大小
                    try:
                        screen_width, screen_height = pyautogui.size()
                        frame = cv2.resize(frame, (screen_width, screen_height))
                    except:
                        # 如果無法獲取螢幕尺寸，使用預設大小
                        frame = cv2.resize(frame, (1280, 720))
                    
                    # 創建全螢幕視窗
                    cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    
                    cv2.imshow(self.window_name, frame)
                    
                    # 檢查按鍵
                    key = cv2.waitKey(30) & 0xFF
                    if key == 27:  # ESC鍵
                        self.is_playing = False
                        break
                        
        except Exception as e:
            print(f"影片播放錯誤: {e}")
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass

def preprocess_image(image_path):
    """預處理圖片以提高人臉檢測成功率"""
    try:
        pil_image = Image.open(image_path)
        
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 調整亮度和對比度
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        image_array = np.array(pil_image)
        return image_array
        
    except Exception as e:
        print(f"預處理圖片 {image_path} 失敗: {str(e)}")
        return None

def load_or_create_encodings(image_paths):
    """載入或創建人臉編碼"""
    if os.path.exists(ENCODINGS_FILE):
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                print("從文件加載預計算的臉部特徵...")
                return pickle.load(f)
        except Exception as e:
            print(f"載入編碼檔案失敗: {e}")
            print("重新計算臉部特徵...")

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
    print(f"\n{'='*50}")
    print(f"處理完成統計:")
    print(f"總圖片數: {len(image_paths)}")
    print(f"成功處理: {valid_images}")
    print(f"失敗數量: {len(failed_images)}")
    print(f"成功率: {valid_images/len(image_paths)*100:.1f}%")

    if valid_images >= 3:
        try:
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump(known_encodings, f)
            print(f"\n✅ 特徵編碼已保存到 {ENCODINGS_FILE}")
        except Exception as e:
            print(f"保存編碼檔案失敗: {e}")
    else:
        print(f"\n❌ 錯誤: 只有 {valid_images} 張有效照片，至少需要3張")

    return known_encodings

def main():
    """主程式"""
    # 圖片路徑列表
    image_paths = [
        "02/03 (1).jpg",
        "02/03 (2).jpg",
        "02/03 (3).jpg",
        "02/03 (4).jpg",
        "02/03 (5).jpg",
        "02/03 (6).jpg",
        "02/03 (7).jpg",
        "02/03 (8).jpg",
        "02/03 (9).jpg",
        "02/03 (10).jpg",
        "02/03 (11).jpg",
        "02/03 (12).jpg",
        "02/03 (13).jpg",
        "02/03 (14).jpg",
        "02/03 (15).jpg",
        "02/03 (16).jpg",
        "02/03 (17).jpg",
        "02/03 (18).jpg",
        "02/03 (19).jpg",
        "02/03 (20).jpg",
        "02/03 (21).jpg",
        "02/03 (22).jpg",
        "02/03 (23).jpg",
        "02/03 (24).jpg",
        "02/03 (25).jpg",
        "02/03 (26).jpg",
        "02/03 (27).jpg",
        "02/03 (28).jpg",
        "02/03 (29).jpg",
        "02/03 (30).jpg",
        "02/03 (31).jpg",
        "02/03 (32).jpg",
        "02/03 (33).jpg",
        "02/03 (34).jpg",
        "02/03 (35).jpg",
        "02/03 (36).jpg",
        "02/03 (37).jpg",
        "02/03 (38).jpg",
        "02/03 (39).jpg",
        "02/03 (40).jpg",
        "02/03 (41).jpg",
        "02/03 (42).jpg",
        "02/03 (43).jpg",
        "02/03 (44).jpg",
        "02/03 (45).jpg",
        "02/03 (46).jpg",
        "02/03 (47).jpg",
        "02/03 (48).jpg",
        "02/03 (49).jpg",
        "02/03 (50).jpg",
        "02/03 (51).jpg",
        "02/03 (52).jpg",
        "02/03 (53).jpg",
        "02/03 (54).jpg",
        "02/03 (55).jpg",
        "02/03 (56).jpg",
        "02/03 (57).jpg",
        "02/03 (58).jpg",
        "02/03 (59).jpg",
        "02/03 (60).jpg",
        "02/03 (61).jpg",
        "02/03 (62).jpg",
        "02/03 (63).jpg",
        "02/03 (64).jpg",
        "02/03 (65).jpg",
        "02/03 (66).jpg",
        
        "02/01 (1).png",
        "02/01 (2).png",
        "02/01 (3).png",
        "02/01 (4).png",
        "02/01 (5).png",
        "02/01 (6).png",
        "02/01 (7).png",
        "02/01 (8).png",
        "02/01 (9).png",
        "02/01 (10).png",
        "02/01 (11).png",
        "02/01 (12).png",
        "02/01 (13).png",
        "02/01 (14).png",
        "02/01 (15).png",
        "02/01 (16).png",
        "02/01 (17).png",
        "02/01 (18).png",
        "02/01 (19).png",
        "02/01 (20).png",
        "02/01 (21).png",
        "02/01 (22).png",
        "02/01 (23).png",
        "02/01 (24).png",
        "02/01 (25).png",
        "02/01 (26).png",
        "02/01 (27).png",  
    ]
    
    # 載入人臉編碼
    print("正在初始化臉部特徵數據庫...")
    known_face_encodings = load_or_create_encodings(image_paths)
    
    if not known_face_encodings or len(known_face_encodings) < 3:
        print("錯誤: 需要至少3張有效朋友照片才能運行！")
        print("請檢查:")
        print("1. 圖片路徑是否正確")
        print("2. 圖片中是否有清楚的人臉")
        print("3. 圖片格式是否支援")
        return

    # 初始化影片播放器
    video_player = VideoPlayer(VIDEO_PATH)
    
    # 初始化攝影機
    print("初始化攝影機...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤：無法開啟攝影機！")
        print("請檢查:")
        print("1. 攝影機是否連接")
        print("2. 是否有其他程式正在使用攝影機")
        print("3. 攝影機驅動是否正常")
        return

    # 設定攝影機解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 主要檢測循環
    print("🚀 朋友檢測系統啟動中...")
    print("按 'q' 退出")
    print("影片播放時按 ESC 可關閉影片")
    
    friend_detected = False
    frame_count = 0
    no_friend_frames = 0
    fps_start_time = time.time()
    DETECTION_INTERVAL = 3

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                time.sleep(0.1)
                continue

            frame_count += 1
            current_time = time.time()

            # 檢測邏輯
            if friend_detected:
                do_detection = (frame_count % 2 == 0)
            else:
                do_detection = (frame_count % DETECTION_INTERVAL == 0)

            if do_detection:
                # 縮小幀以提高處理速度
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # 檢測人臉
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                friend_found_this_frame = False
                
                if face_locations:
                    # 轉換座標到原始尺寸
                    face_locations = [(int(top * 2), int(right * 2), int(bottom * 2), int(left * 2))
                                    for (top, right, bottom, left) in face_locations]
                    
                    # 獲取人臉編碼
                    face_encodings = face_recognition.face_encodings(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        known_face_locations=face_locations
                    )
                    
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # 計算距離
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if len(face_distances) > 0:
                            min_distance = min(face_distances)
                            is_friend = min_distance < CONFIDENCE_THRESHOLD
                            
                            if is_friend:
                                friend_found_this_frame = True
                                
                                # 計算臉部中心距離
                                face_center = ((left + right) // 2, (top + bottom) // 2)
                                frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                                distance = np.sqrt((face_center[0] - frame_center[0]) ** 2 + 
                                                 (face_center[1] - frame_center[1]) ** 2)
                                
                                if distance < TRIGGER_DISTANCE and not video_player.is_playing:
                                    video_player.play()
                                    friend_detected = True
                                    print(f"🎉 檢測到 {FRIEND_NAME}！開始播放影片")
                                
                                if distance < TRIGGER_DISTANCE:
                                    friend_detected = True
                                    no_friend_frames = 0
                                
                                # 繪製綠色框
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                info_text = f"{FRIEND_NAME} ({min_distance:.3f})"
                                frame = cv2_puttext_chinese(frame, info_text, (left, top - 30), 15, (0, 255, 0))
                            else:
                                # 繪製紅色框
                                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                info_text = f"Unknown ({min_distance:.3f})"
                                frame = cv2_puttext_chinese(frame, info_text, (left, top - 30), 15, (0, 0, 255))
                
                # 處理朋友離開邏輯
                if friend_detected:
                    if not friend_found_this_frame:
                        no_friend_frames += 1
                        if no_friend_frames >= NO_FRIEND_FRAMES_THRESHOLD:
                            friend_detected = False
                            no_friend_frames = 0
                            print(f"👋 {FRIEND_NAME} 已離開")
                    else:
                        no_friend_frames = 0

            # 顯示狀態
            status = "檢測中" if friend_detected else "監控中"
            video_status = "播放中" if video_player.is_playing else "待機"
            
            status_text = f"狀態: {status} | 影片: {video_status}"
            frame = cv2_puttext_chinese(frame, status_text, (10, 30), 20, (255, 255, 255))
            
            # 計算FPS
            if frame_count % 30 == 0:
                fps = 30 / (current_time - fps_start_time + 0.001)
                fps_start_time = current_time
                fps_text = f"FPS: {int(fps)}"
                frame = cv2_puttext_chinese(frame, fps_text, (frame.shape[1] - 120, 30), 15, (255, 255, 255))

            cv2.imshow("Friend Detector", frame)
            
            # 檢查退出鍵
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n程式被用戶中斷")
    except Exception as e:
        print(f"程式執行錯誤: {e}")
    finally:
        # 清理資源
        print("正在清理資源...")
        if video_player.is_playing:
            video_player.stop()
        cap.release()
        cv2.destroyAllWindows()
        print("程式已安全關閉")

if __name__ == "__main__":
    main()