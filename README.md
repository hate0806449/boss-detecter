# 朋友檢測系統 Friend Detection System

這是一個基於人臉識別的朋友檢測系統，當檢測到特定朋友時會自動播放影片。

## 檔案結構

```
friend-detection-system/
├── main.py                     # 主程式入口
├── config.py                   # 設定檔
├── utils.py                    # 工具函數
├── video_player.py             # 影片播放器
├── face_recognition_handler.py # 人臉識別處理
├── friend_detector.py          # 朋友檢測器
├── requirements.txt            # 依賴套件
├── README.md                   # 使用說明
├── known_face_encodings.pkl    # 人臉特徵編碼檔案 (自動生成)
├── update.mp4                  # 要播放的影片檔案
└── 02/                         # 朋友照片資料夾
    ├── 03 (1).jpg
    ├── 03 (2).jpg
    └── ...
```

## 安裝與設定

### 1. 安裝依賴套件

```bash
pip install -r requirements.txt
```

### 2. 準備朋友照片

將朋友的照片放在 `02/` 資料夾中，支援 `.jpg` 和 `.png` 格式。
- 建議至少準備 10-20 張不同角度的照片
- 照片中人臉要清楚可見
- 避免過暗或過亮的照片

### 3. 準備影片檔案

將要播放的影片命名為 `update.mp4` 並放在專案根目錄。

### 4. 調整設定

編輯 `config.py` 檔案調整參數：

- `FRIEND_NAME`: 朋友的名字
- `CONFIDENCE_THRESHOLD`: 人臉識別信心度閾值 (越小越嚴格)
- `TRIGGER_DISTANCE`: 觸發播放的距離
- `VIDEO_PATH`: 影片檔案路徑

## 使用方法

1. 執行主程式：
```bash
python main.py
```

2. 首次執行會自動分析朋友照片並建立人臉特徵資料庫

3. 系統啟動後會開啟攝影機視窗，當檢測到朋友時會自動播放影片

4. 按 `q` 鍵退出程式，影片播放時按 `ESC` 鍵關閉影片

## 功能說明

### 核心功能
- **人臉檢測**: 即時檢測攝影機畫面中的人臉
- **朋友識別**: 比對檢測到的人臉與預設的朋友照片
- **自動播放**: 當朋友靠近時自動播放指定影片
- **全螢幕播放**: 影片以全螢幕模式播放

### 界面顯示
- 綠色框: 檢測到的朋友
- 紅色框: 檢測到的陌生人
- 狀態資訊: 顯示當前檢測狀態和影片播放狀態
- FPS 顯示: 顯示當前處理速度

## 系統要求

- Python 3.7+
- 攝影機 (webcam)
- Windows/Linux/macOS

## 故障排除

### 攝影機無法開啟
- 檢查攝影機是否正確連接
- 確認沒有其他程式正在使用攝影機
- 更新攝影機驅動程式

### 人臉識別效果不佳
- 增加更多不同角度的朋友照片
- 調整 `CONFIDENCE_THRESHOLD` 參數
- 確保照片品質良好，人臉清楚

### 影片無法播放
- 檢查影片檔案是否存在且格式正確
- 確認影片編解碼器已安裝

## 自訂設定

您可以透過修改 `config.py` 來自訂系統行為：

```python
# 朋友名稱
FRIEND_NAME = "Your Friend"

# 識別敏感度 (0.0-1.0，越小越嚴格)
CONFIDENCE_THRESHOLD = 0.37

# 觸發距離 (像素)
TRIGGER_DISTANCE = 180

# 影片檔案路徑
VIDEO_PATH = "update.mp4"
```

## 注意事項

- 首次執行需要較長時間來分析照片
- 系統會自動保存人臉特徵到 `known_face_encodings.pkl`
- 建議在光線充足的環境中使用
- 為保護隱私，請妥善保管人臉特徵檔案