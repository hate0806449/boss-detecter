import cv2
import time
from config import IMAGE_PATHS, VIDEO_PATH, CAMERA_WIDTH, CAMERA_HEIGHT
from face_recognition_handler import FaceRecognitionHandler
from video_player import VideoPlayer
from friend_detector import FriendDetector


def initialize_camera():
    """
    初始化攝影機
    
    Returns:
        攝影機物件，失敗時返回 None
    """
    print("初始化攝影機...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("錯誤：無法開啟攝影機！")
        print("請檢查:")
        print("1. 攝影機是否連接")
        print("2. 是否有其他程式正在使用攝影機")
        print("3. 攝影機驅動是否正常")
        return None

    # 設定攝影機解析度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    return cap

def main():
    """主程式"""
    print("🚀 朋友檢測系統啟動中...")
    
    # 初始化人臉識別處理器
    print("正在初始化臉部特徵數據庫...")
    face_handler = FaceRecognitionHandler()
    known_face_encodings = face_handler.load_or_create_encodings(IMAGE_PATHS)
    
    # 檢查編碼數量
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
    cap = initialize_camera()
    if cap is None:
        return
    
    # 初始化朋友檢測器
    detector = FriendDetector(face_handler, video_player)
    
    # 主要檢測循環
    print("系統已啟動！")
    print("按 'q' 退出")
    print("影片播放時按 ESC 可關閉影片")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝影機畫面")
                time.sleep(0.1)
                continue

            # 處理畫面
            processed_frame = detector.process_frame(frame)
            
            # 顯示畫面
            cv2.imshow("Friend Detector", processed_frame)
            
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