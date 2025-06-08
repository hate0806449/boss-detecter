import cv2
import os
import threading
import pyautogui


class VideoPlayer:
    """影片播放器類別"""
    
    def __init__(self, video_path):
        """
        初始化影片播放器
        
        Args:
            video_path: 影片檔案路径
        """
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
        """開始播放影片"""
        if not self.video_exists:
            print("無法播放影片：檔案不存在")
            return
            
        if not self.is_playing:
            self.is_playing = True
            self.thread = threading.Thread(target=self._play_loop)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        """停止播放影片"""
        self.is_playing = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        try:
            cv2.destroyWindow(self.window_name)
        except:
            pass
    
    def _play_loop(self):
        """影片播放循環（私有方法）"""
        if not self.video_exists:
            return
            
        try:
            while self.is_playing:
                # 重置影片到開頭
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
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
                    
                    # 檢查按鍵 (ESC 鍵退出)
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