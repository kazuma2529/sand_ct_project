import json, cv2, numpy as np
from pathlib import Path

input_dir  = Path("data/train/annotations")
output_dir = Path("data/train/masks")
output_dir.mkdir(exist_ok=True)

for json_file in input_dir.glob("*.json"):
    with open(json_file, "r") as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data["shapes"]:
        pts = np.array(shape["points"], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=255)

    out_path = output_dir / (json_file.stem + ".png")
    cv2.imwrite(str(out_path), mask)
    print("✔", out_path)

print("✅ すべてのマスク画像が生成されました！")
