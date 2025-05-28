import cv2
import numpy as np
import pyautogui
import keyboard
import time
import threading
import face_recognition
from PIL import Image, ImageDraw, ImageFont
import pickle
import os

# 設定
FRIEND_NAME = "Your Friend"  # 朋友的名字
WORK_MODE_KEY = "alt+tab"  # 切換到工作模式的快捷鍵
CONFIDENCE_THRESHOLD = 0.35  # 降低閾值，提高識別嚴格度（距離越小越相似）
TRIGGER_DISTANCE = 180  # 觸發距離（像素）
ENCODINGS_FILE = "known_face_encodings.pkl"  # 特徵編碼保存文件
DETECTION_INTERVAL = 3  # 每隔幾幀進行一次完整檢測（降低CPU負載）
VIDEO_PATH = "update.mp4"  # 要播放的影片路徑
NO_FRIEND_FRAMES_THRESHOLD = 10  # 連續多少幀沒檢測到朋友才確認朋友離開

# 中文文字繪製函數
def cv2_puttext_chinese(img, text, position, font_size, color):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font = ImageFont.truetype("msyh.ttc", font_size)  # 微軟雅黑
    except:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 影片播放器類（支援置頂功能）
class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.is_playing = False
        self.window_name = "Video Player"
        self.current_frame = None
        self.thread = None
    
    def play(self):
        if not self.is_playing:
            self.is_playing = True
            self.thread = threading.Thread(target=self._play_loop)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        self.is_playing = False
        if self.thread:
            self.thread.join()
        cv2.destroyWindow(self.window_name)
    
    def _play_loop(self):
        while self.is_playing:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到影片開頭
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            screen_width, screen_height = pyautogui.size()
            frame = cv2.resize(frame, (screen_width, screen_height))
            
            # 創建視窗並設置為全螢幕和置頂
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # 使用Windows API將視窗置頂（如果是Windows系統）
            try:
                import win32gui
                import win32con
                # 尋找OpenCV視窗的句柄
                def enum_windows_callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        window_text = win32gui.GetWindowText(hwnd)
                        if self.window_name in window_text:
                            windows.append(hwnd)
                    return True
                
                windows = []
                win32gui.EnumWindows(enum_windows_callback, windows)
                
                # 將視窗設為最頂層
                for hwnd in windows:
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, 
                                        win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                    break
                    
            except ImportError:
                # 如果沒有安裝win32gui，使用備用方法
                print("提示：安裝 pywin32 套件可獲得更好的置頂效果")
                # 備用：多次呼叫 imshow 並設置視窗屬性
                pass
            
            while ret and self.is_playing:
                cv2.imshow(self.window_name, frame)
                
                # 確保視窗保持在最頂層（備用方法）
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
                
                # 只檢查 ESC 鍵 (27)，按下時停止播放
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC 鍵
                    self.is_playing = False
                    # 恢復視窗正常狀態後關閉
                    try:
                        import win32gui
                        import win32con
                        windows = []
                        win32gui.EnumWindows(enum_windows_callback, windows)
                        for hwnd in windows:
                            win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, 
                                                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
                            break
                    except:
                        pass
                    cv2.destroyWindow(self.window_name)
                    return
                    
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, (screen_width, screen_height))

# 1. 預先載入或創建臉部特徵編碼（改進版）
def load_or_create_encodings(image_paths):
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            print("從文件加載預計算的臉部特徵...")
            return pickle.load(f)

    print("首次運行，計算臉部特徵...")
    known_encodings = []
    valid_images = 0

    for path in image_paths:
        try:
            # 使用face_recognition內建函數處理圖片（自動旋轉）
            image = face_recognition.load_image_file(path)

            # 先檢測人臉位置（提高準確度）
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                print(f"跳過 {path} - 未檢測到人臉")
                continue

            # 只取最大的人臉（假設每張照片只有目標人物）
            encodings = face_recognition.face_encodings(
                image, known_face_locations=[face_locations[0]]
            )
            if encodings:
                known_encodings.append(encodings[0])
                valid_images += 1
                print(f"成功處理: {path}")
        except Exception as e:
            print(f"處理照片 {path} 時發生錯誤: {str(e)}")

    if valid_images >= 3:  # 至少需要3張有效照片
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(known_encodings, f)
        print(f"保存 {valid_images} 個臉部特徵到 {ENCODINGS_FILE}")
    else:
        print(f"錯誤: 只有 {valid_images} 張有效照片，至少需要3張")

    return known_encodings

# 替換成你朋友的多張照片路徑
image_paths = [
    "02/02 (1).jpg",
    "02/02 (2).jpg",
    "02/02 (3).jpg",
    "02/02 (4).jpg",
    "02/02 (5).jpg",
    "02/02 (6).jpg",
    "02/02 (7).jpg",
    "02/02 (8).jpg",
    "02/02 (9).jpg",
    "02/02 (10).jpg",
    "02/02 (11).jpg",
    "02/02 (12).jpg",
    "02/02 (13).jpg",
    "02/02 (14).jpg",
    "02/02 (15).jpg",
    "02/02 (16).jpg",
    "02/02 (17).jpg",
    "02/02 (18).jpg",
    "02/02 (19).jpg",
    "02/02 (20).jpg",
    "02/02 (21).jpg",
    "02/02 (22).jpg",
    "02/02 (23).jpg",
    "02/02 (24).jpg",
    "02/02 (25).jpg",
    "02/02 (26).jpg",
    "02/02 (27).jpg",
    "02/02 (28).jpg",
    "02/02 (29).jpg",
    "02/02 (30).jpg",
    "02/02 (31).jpg",
    "02/02 (32).jpg",
    "02/02 (33).jpg",
    "02/02 (34).jpg",
    "02/02 (35).jpg",
    "02/02 (36).jpg",
    "02/02 (37).jpg",
    "02/02 (38).jpg",
    "02/02 (39).jpg",
    "02/02 (40).jpg",
    "02/02 (41).jpg",
    "02/02 (42).jpg",
    "02/02 (43).jpg",
    "02/02 (44).jpg",
    "02/02 (45).jpg",
    "02/02 (46).jpg",
    "02/02 (47).jpg",
    "02/02 (48).jpg",
    "02/02 (49).jpg",
    "02/02 (50).jpg",
    "02/02 (51).jpg",
    "02/02 (52).jpg",
    "02/02 (53).jpg",
    "02/02 (54).jpg",
    "02/02 (55).jpg",
    "02/02 (56).jpg",
    "02/02 (57).jpg",
    "02/02 (58).jpg",
    "02/02 (59).jpg",
    "02/02 (60).jpg",
    "02/02 (61).jpg",
    "02/02 (62).jpg",
    "02/02 (63).jpg",
    "02/02 (64).jpg",
    "02/02 (65).jpg",
    "02/02 (66).jpg",
]

# 載入所有已知臉部特徵（改進載入邏輯）
print("正在初始化臉部特徵數據庫...")
known_face_encodings = load_or_create_encodings(image_paths)
if not known_face_encodings or len(known_face_encodings) < 3:
    print("錯誤: 需要至少3張有效朋友照片才能運行！")
    exit()

# 初始化影片播放器
video_player = VideoPlayer(VIDEO_PATH)

# 初始化鏡頭（增加重試邏輯）
cap = None
for _ in range(3):
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        break
    print("鏡頭初始化失敗，重試中...")
    time.sleep(1)

if not cap or not cap.isOpened():
    print("錯誤：無法開啟鏡頭！")
    exit()

# 設定鏡頭解析度（兼容更多設備）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 主迴圈（優化偵測邏輯）
friend_detected = False
frame_count = 0
last_detection_time = 0
no_friend_frames = 0  # 連續沒檢測到朋友的幀數計數器
fps_start_time = time.time()  # FPS計算用的時間戳記
print("朋友檢測系統啟動中... 按 'q' 退出")
print("提示：影片播放時會置於螢幕最頂層，按 ESC 可關閉影片")

while True:
    ret, frame = cap.read()
    if not ret:
        print("無法讀取鏡頭畫面，重試中...")
        time.sleep(1)
        continue

    frame_count += 1
    current_time = time.time()

    # 修正檢測邏輯：正常監控時使用間隔檢測，檢測到朋友時提高檢測頻率
    if friend_detected:
        # 朋友已檢測到時，每幀都檢測以確保實時反應
        do_full_detection = True
    else:
        # 正常監控時，使用間隔檢測降低CPU負載
        do_full_detection = (frame_count % DETECTION_INTERVAL == 0)

    if do_full_detection:
        last_detection_time = current_time

        # 縮小畫面提高處理速度（保持比例）
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # 使用HOG模型檢測人臉（平衡速度與準確度）
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        friend_found_this_frame = False

        if face_locations:
            # 轉換座標到原始尺寸
            face_locations = [
                (top * 2, right * 2, bottom * 2, left * 2)
                for (top, right, bottom, left) in face_locations
            ]

            # 獲取臉部特徵編碼
            face_encodings = face_recognition.face_encodings(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                known_face_locations=face_locations,
            )

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                # 計算與已知臉部的距離
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                min_distance = min(face_distances)
                best_match_index = np.argmin(face_distances)
                similarity = (1 - min_distance) * 100

                # 更嚴格的判斷邏輯
                is_friend = min_distance < CONFIDENCE_THRESHOLD
                
                # 調試資訊 - 顯示詳細的識別數據
                debug_text = f"距離: {min_distance:.3f}, 相似度: {similarity:.1f}%"
                print(f"檢測到人臉 - {debug_text}, 是朋友: {is_friend}")

                if is_friend:
                    friend_found_this_frame = True
                    name = FRIEND_NAME
                    color = (0, 255, 0)  # 綠色框

                    # 計算臉部中心與畫面中心的距離
                    face_center = ((left + right) // 2, (top + bottom) // 2)
                    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                    distance = np.sqrt(
                        (face_center[0] - frame_center[0]) ** 2
                        + (face_center[1] - frame_center[1]) ** 2
                    )

                    # 觸發邏輯：只有當朋友進入偵測範圍且影片未播放時才啟動
                    if distance < TRIGGER_DISTANCE and not video_player.is_playing:
                        video_player.play()
                        friend_detected = True
                        print(f"檢測到 {FRIEND_NAME}，開始播放置頂影片")

                    # 如果朋友在範圍內，重置狀態
                    if distance < TRIGGER_DISTANCE:
                        friend_detected = True
                        no_friend_frames = 0  # 重置計數器

                    # 繪製標記
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    info_text = f"{name} 距離:{min_distance:.3f} 位置:{int(distance)}px"
                    frame = cv2_puttext_chinese(
                        frame, info_text, (left, top - 30), 15, color
                    )
                else:
                    # 非朋友用紅色標記，顯示詳細資訊
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    info_text = f"未知 距離:{min_distance:.3f} ({similarity:.1f}%)"
                    frame = cv2_puttext_chinese(
                        frame, info_text, (left, top - 30), 15, (0, 0, 255)
                    )

        # 處理朋友離開的邏輯（但不關閉影片）
        if friend_detected:
            if not friend_found_this_frame:
                no_friend_frames += 1
                # 連續多幀沒檢測到朋友才確認離開，避免偶發的檢測失敗
                if no_friend_frames >= NO_FRIEND_FRAMES_THRESHOLD:
                    friend_detected = False
                    no_friend_frames = 0
                    # 注意：不自動關閉影片，只重置檢測狀態
                    print(f"{FRIEND_NAME} 已離開，恢復監控模式（影片繼續播放）")
            else:
                no_friend_frames = 0  # 重置計數器

    # 顯示狀態（優化顯示）
    video_status = "置頂播放中" if video_player.is_playing else "未播放"
    if friend_detected:
        status_text = f"狀態: {FRIEND_NAME} 出現！影片: {video_status}"
        status_color = (0, 255, 0)
    else:
        status_text = f"監控中... 影片: {video_status}"
        status_color = (255, 255, 255)
    
    frame = cv2_puttext_chinese(frame, status_text, (10, 30), 20, status_color)

    # 計算並顯示實際FPS
    if frame_count % 30 == 0:  # 每30幀計算一次FPS
        fps = 30 / (current_time - fps_start_time + 0.001)
        fps_start_time = current_time
    else:
        fps = 30 / (current_time - fps_start_time + 0.001) if frame_count > 30 else 0

    fps_text = f"FPS: {int(fps)}"
    frame = cv2_puttext_chinese(
        frame, fps_text, (frame.shape[1] - 150, 30), 20, (255, 255, 255)
    )
    
    # 顯示當前檢測參數
    param_text = f"閾值: {CONFIDENCE_THRESHOLD}"
    frame = cv2_puttext_chinese(
        frame, param_text, (frame.shape[1] - 200, 60), 15, (255, 255, 0)
    )

    # 顯示檢測模式和影片控制提示
    mode_text = "高頻檢測" if friend_detected else f"間隔檢測({DETECTION_INTERVAL}幀)"
    frame = cv2_puttext_chinese(
        frame, mode_text, (10, frame.shape[0] - 90), 15, (200, 200, 200)
    )
    
    # 影片控制提示
    if video_player.is_playing:
        control_text = "影片已置頂顯示 - 按ESC關閉"
        frame = cv2_puttext_chinese(
            frame, control_text, (10, frame.shape[0] - 60), 15, (0, 255, 255)
        )
    
    # 置頂功能說明
    topmost_info = "影片將覆蓋所有視窗"
    frame = cv2_puttext_chinese(
        frame, topmost_info, (10, frame.shape[0] - 30), 15, (255, 165, 0)
    )

    cv2.imshow("Friend Detector - 置頂影片播放", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        if video_player.is_playing:
            video_player.stop()
        break

# 清理資源
cap.release()
cv2.destroyAllWindows()
print("朋友檢測系統已關閉")