import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
import sys
from tqdm import tqdm
import logging

# 親ディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import UNet

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('prediction.log'),
        logging.StreamHandler()
    ]
)

def predict_mask(model, image_path, device):
    """画像から粒子のマスクを予測する"""
    try:
        # 画像の読み込みと前処理
        image = Image.open(image_path).convert('L')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image_tensor = transform(image).unsqueeze(0)
        
        # 予測
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            output = model(image_tensor)
            predicted_mask = (output > 0.5).float()
            
        # numpy配列に変換
        predicted_mask = predicted_mask.cpu().squeeze().numpy()
        return predicted_mask
    
    except Exception as e:
        logging.error(f"予測中にエラーが発生しました（{image_path}）: {str(e)}")
        raise

def process_batch(model, image_paths, device, output_dir):
    """バッチ処理で画像を予測"""
    for image_path in tqdm(image_paths, desc="予測中"):
        try:
            image_name = os.path.basename(image_path)
            predicted_mask = predict_mask(model, image_path, device)
            
            # マスクの保存
            mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
            output_path = os.path.join(output_dir, f'pred_{image_name}')
            mask_image.save(output_path)
            
            logging.info(f'処理完了: {image_name}')
            
        except Exception as e:
            logging.error(f"画像の処理中にエラーが発生しました（{image_path}）: {str(e)}")
            continue

def main():
    try:
        # デバイスの設定
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"使用デバイス: {device}")
        
        # モデルの読み込み
        model = UNet(in_channels=1, out_channels=1).to(device)
        checkpoint = torch.load('best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("モデルを読み込みました")
        
        # 推論対象の画像ディレクトリ
        test_dir = 'data/test/images'
        output_dir = 'data/test/predicted_masks'
        os.makedirs(output_dir, exist_ok=True)
        
        # 画像パスのリストを作成
        image_paths = [os.path.join(test_dir, f) for f in os.listdir(test_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_paths:
            logging.warning(f"テストディレクトリに画像が見つかりません: {test_dir}")
            return
        
        # バッチ処理で予測を実行
        process_batch(model, image_paths, device, output_dir)
        
        logging.info("全ての画像の処理が完了しました")
        
    except Exception as e:
        logging.error(f"予測プロセス中にエラーが発生しました: {str(e)}")
        raise

if __name__ == '__main__':
    main() 