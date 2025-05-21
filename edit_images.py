import os
from pathlib import Path
import cv2
import numpy as np

in_dir  = Path("data/train/images")
out_dir = Path("data/train/images_enhanced")
out_dir.mkdir(parents=True, exist_ok=True)

# ===== 調整量 =====
CLAHE_CLIP        = 4.0   # コントラスト増強の強さ（大きいほど強調）
SHADOW_THRESHOLD  = 90    # 0〜255。暗い領域の判定値
SHADOW_DARKENING  = 0.4   # 0.0〜1.0。暗い領域をどれだけ暗くするか

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=(8, 8))

for img_file in in_dir.glob("*.[pj][pn]g"):
    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    # 1) コントラスト（局所ヒストグラム補正）
    img = clahe.apply(img)

    # 2) シャドー -100 相当
    mask    = img < SHADOW_THRESHOLD     # True: シャドー領域
    darker  = img.copy()
    darker[mask] = (darker[mask] * SHADOW_DARKENING).astype(np.uint8)

    cv2.imwrite(str(out_dir / img_file.name), darker)

print("✅ 完了 →", out_dir)
