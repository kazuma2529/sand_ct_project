import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import logging

class ParticleDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # ディレクトリの存在確認
        if not os.path.exists(image_dir):
            raise ValueError(f"画像ディレクトリが存在しません: {image_dir}")
        if not os.path.exists(mask_dir):
            raise ValueError(f"マスクディレクトリが存在しません: {mask_dir}")
            
        self.images = sorted(os.listdir(image_dir))
        
        # マスク画像の検証
        self._validate_masks()
    
    def _validate_masks(self):
        """マスク画像の検証を行う"""
        for img_name in self.images:
            mask_path = os.path.join(self.mask_dir, img_name)
            if not os.path.exists(mask_path):
                logging.warning(f"マスクファイルが存在しません: {mask_path}")
                continue
                
            # マスクの値を検証
            mask = Image.open(mask_path).convert('L')
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            
            if not np.all(np.isin(unique_values, [0, 255])):
                logging.warning(f"マスク画像 {mask_path} に0と255以外の値が含まれています: {unique_values}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.image_dir, self.images[idx])
            mask_path = os.path.join(self.mask_dir, self.images[idx])
            
            # グレースケールとして読み込み
            image = Image.open(img_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            
            # サイズの一致確認
            if image.size != mask.size:
                raise ValueError(f"画像とマスクのサイズが一致しません: {img_path}")
            
            # PIL画像をTensorに変換
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
            
            # マスクの値を0または1に正規化
            mask = (mask > 0.5).float()
            
            if self.transform:
                image = self.transform(image)
                
            return image, mask
            
        except Exception as e:
            logging.error(f"エラーが発生しました（{self.images[idx]}）: {str(e)}")
            raise

def get_transforms():
    """データ拡張のための変換を定義"""
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform 