import cv2
import threading
import pyautogui
import time
import os

class VideoPlayer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.playing = False
        self.thread = None
        self.window_name = "Friend Video"
        self.maximized = False

    def _play_loop(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("❌ 無法打開影片")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, 0, 0)

        # 僅最大化一次
        if not self.maximized:
            try:
                pyautogui.getWindowsWithTitle(self.window_name)[0].maximize()
                self.maximized = True
            except IndexError:
                print("⚠️ 無法最大化視窗（找不到視窗）")

        while self.playing:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(30) & 0xFF == 27:  # ESC 離開
                break

        cap.release()
        cv2.destroyWindow(self.window_name)
        self.playing = False
        self.thread = None

    def play(self):
        if not self.playing:
            self.playing = True
            self.thread = threading.Thread(target=self._play_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.playing = False
