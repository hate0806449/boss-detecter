import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont


def cv2_puttext_chinese(img, text, position, font_size, color):
    """
    在OpenCV圖像上繪製中文文字
    
    Args:
        img: OpenCV圖像
        text: 要繪製的文字
        position: 文字位置 (x, y)
        font_size: 字體大小
        color: 文字顏色 (B, G, R)
    
    Returns:
        繪製文字後的圖像
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 嘗試載入字體
    try:
        font = ImageFont.truetype("msyh.ttc", font_size)  # Windows 微軟雅黑
    except:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)  # 備用字體
        except:
            font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def preprocess_image(image_path):
    """
    預處理圖片以提高人臉檢測成功率
    
    Args:
        image_path: 圖片路径
    
    Returns:
        預處理後的圖像陣列，失敗時返回 None
    """
    try:
        pil_image = Image.open(image_path)
        
        # 確保圖片為 RGB 格式
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # 調整亮度和對比度
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        return np.array(pil_image)
        
    except Exception as e:
        print(f"預處理圖片 {image_path} 失敗: {str(e)}")
        return None


def calculate_face_center_distance(face_location, frame_shape):
    """
    計算人臉中心與畫面中心的距離
    
    Args:
        face_location: 人臉位置 (top, right, bottom, left)
        frame_shape: 畫面尺寸 (height, width, channels)
    
    Returns:
        距離值
    """
    top, right, bottom, left = face_location
    face_center = ((left + right) // 2, (top + bottom) // 2)
    frame_center = (frame_shape[1] // 2, frame_shape[0] // 2)
    
    distance = np.sqrt((face_center[0] - frame_center[0]) ** 2 + 
                      (face_center[1] - frame_center[1]) ** 2)
    return distance