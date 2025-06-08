import cv2
import numpy as np
import time
from utils import cv2_puttext_chinese, calculate_face_center_distance
from config import (
    FRIEND_NAME, CONFIDENCE_THRESHOLD, TRIGGER_DISTANCE, 
    NO_FRIEND_FRAMES_THRESHOLD, DETECTION_INTERVAL, FPS_UPDATE_INTERVAL
)


class FriendDetector:
    """朋友檢測器"""
    
    def __init__(self, face_handler, video_player):
        """
        初始化朋友檢測器
        
        Args:
            face_handler: 人臉識別處理器
            video_player: 影片播放器
        """
        self.face_handler = face_handler
        self.video_player = video_player
        
        # 檢測狀態
        self.friend_detected = False
        self.no_friend_frames = 0
        
        # 計數器
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def process_frame(self, frame):
        """
        處理單一畫面
        
        Args:
            frame: 攝影機畫面
        
        Returns:
            處理後的畫面
        """
        self.frame_count += 1
        current_time = time.time()
        
        # 決定是否執行檢測
        should_detect = self._should_detect_this_frame()
        
        if should_detect:
            self._detect_faces_in_frame(frame)
        
        # 繪製界面資訊
        frame = self._draw_interface(frame)
        
        # 更新 FPS
        if self.frame_count % FPS_UPDATE_INTERVAL == 0:
            self._update_fps(current_time)
        
        return frame
    
    def _should_detect_this_frame(self):
        """判斷是否應該在此幀執行檢測"""
        if self.friend_detected:
            return (self.frame_count % 2 == 0)  # 檢測到朋友時更頻繁檢測
        else:
            return (self.frame_count % DETECTION_INTERVAL == 0)  # 未檢測到時降低頻率
    
    def _detect_faces_in_frame(self, frame):
        """在畫面中檢測人臉"""
        # 縮小幀以提高處理速度
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # 檢測人臉位置
        face_locations = self.face_handler.recognize_faces(rgb_small_frame)[0]
        
        friend_found_this_frame = False
        
        if face_locations:
            # 轉換座標到原始尺寸
            face_locations = [(int(top * 2), int(right * 2), int(bottom * 2), int(left * 2))
                            for (top, right, bottom, left) in face_locations]
            
            # 獲取完整解析度的人臉編碼
            _, _, face_distances_list = self.face_handler.recognize_faces(frame)
            
            for i, (face_location, face_distances) in enumerate(zip(face_locations, face_distances_list)):
                if len(face_distances) > 0:
                    min_distance = min(face_distances)
                    is_friend = min_distance < CONFIDENCE_THRESHOLD
                    
                    # 繪製人臉框和資訊
                    self._draw_face_info(frame, face_location, min_distance, is_friend)
                    
                    if is_friend:
                        friend_found_this_frame = True
                        self._handle_friend_detection(face_location, frame.shape)
        
        # 處理朋友離開邏輯
        self._handle_friend_absence(friend_found_this_frame)
    
    def _draw_face_info(self, frame, face_location, distance, is_friend):
        """繪製人臉框和資訊"""
        top, right, bottom, left = face_location
        
        if is_friend:
            # 朋友 - 綠色框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            info_text = f"{FRIEND_NAME} ({distance:.3f})"
            color = (0, 255, 0)
        else:
            # 陌生人 - 紅色框
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            info_text = f"Unknown ({distance:.3f})"
            color = (0, 0, 255)
        
        frame = cv2_puttext_chinese(frame, info_text, (left, top - 30), 15, color)
        return frame
    
    def _handle_friend_detection(self, face_location, frame_shape):
        """處理檢測到朋友的邏輯"""
        # 計算臉部中心距離
        distance = calculate_face_center_distance(face_location, frame_shape)
        
        # 如果朋友在觸發距離內且影片未播放，開始播放
        if distance < TRIGGER_DISTANCE and not self.video_player.is_playing:
            self.video_player.play()
            print(f"🎉 檢測到 {FRIEND_NAME}！開始播放影片")
        
        # 更新檢測狀態
        if distance < TRIGGER_DISTANCE:
            self.friend_detected = True
            self.no_friend_frames = 0
    
    def _handle_friend_absence(self, friend_found_this_frame):
        """處理朋友離開的邏輯"""
        if self.friend_detected:
            if not friend_found_this_frame:
                self.no_friend_frames += 1
                if self.no_friend_frames >= NO_FRIEND_FRAMES_THRESHOLD:
                    self.friend_detected = False
                    self.no_friend_frames = 0
                    print(f"👋 {FRIEND_NAME} 已離開")
            else:
                self.no_friend_frames = 0
    
    def _draw_interface(self, frame):
        """繪製界面資訊"""
        # 狀態資訊
        status = "檢測中" if self.friend_detected else "監控中"
        video_status = "播放中" if self.video_player.is_playing else "待機"
        status_text = f"狀態: {status} | 影片: {video_status}"
        frame = cv2_puttext_chinese(frame, status_text, (10, 30), 20, (255, 255, 255))
        
        # FPS 顯示
        if self.current_fps > 0:
            fps_text = f"FPS: {int(self.current_fps)}"
            frame = cv2_puttext_chinese(frame, fps_text, (frame.shape[1] - 120, 30), 15, (255, 255, 255))
        
        return frame
    
    def _update_fps(self, current_time):
        """更新 FPS 計算"""
        time_diff = current_time - self.fps_start_time + 0.001
        self.current_fps = FPS_UPDATE_INTERVAL / time_diff
        self.fps_start_time = current_time