import os
import numpy as np
from PIL import Image
import pandas as pd
from scipy import ndimage
import logging
from tqdm import tqdm

# ロギングの設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)

def load_mask_stack(mask_dir):
    """マスク画像を3Dスタックとして読み込む"""
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.startswith('pred_')])
    if not mask_files:
        raise ValueError(f"マスクファイルが見つかりません: {mask_dir}")
    
    # 最初のマスクを読み込んでサイズを取得
    first_mask = np.array(Image.open(os.path.join(mask_dir, mask_files[0])))
    stack = np.zeros((len(mask_files), *first_mask.shape), dtype=np.uint8)
    
    for i, mask_file in enumerate(tqdm(mask_files, desc="マスクを読み込み中")):
        mask = np.array(Image.open(os.path.join(mask_dir, mask_file)))
        stack[i] = mask
    
    return stack

def label_particles(mask_stack):
    """3Dマスクスタックから粒子をラベル付け"""
    # 2値化
    binary_stack = (mask_stack > 127).astype(np.uint8)
    
    # 3Dラベル付け
    labeled_stack, num_particles = ndimage.label(binary_stack)
    logging.info(f"検出された粒子数: {num_particles}")
    
    return labeled_stack, num_particles

def analyze_contacts(labeled_stack):
    """粒子間の接触を分析（接触面積を考慮）"""
    num_particles = np.max(labeled_stack)
    contact_matrix = np.zeros((num_particles + 1, num_particles + 1), dtype=int)
    contact_area_matrix = np.zeros((num_particles + 1, num_particles + 1), dtype=float)
    
    # 各方向の接触をチェック
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    for z in tqdm(range(labeled_stack.shape[0]), desc="接触を分析中"):
        for y in range(labeled_stack.shape[1]):
            for x in range(labeled_stack.shape[2]):
                current_label = labeled_stack[z, y, x]
                if current_label == 0:
                    continue
                
                # 各方向の隣接ピクセルをチェック
                for dz, dy, dx in directions:
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < labeled_stack.shape[0] and
                        0 <= ny < labeled_stack.shape[1] and
                        0 <= nx < labeled_stack.shape[2]):
                        neighbor_label = labeled_stack[nz, ny, nx]
                        if neighbor_label > 0 and neighbor_label != current_label:
                            contact_matrix[current_label, neighbor_label] += 1
                            # 接触面積を計算（ピクセル単位）
                            contact_area_matrix[current_label, neighbor_label] += 1
    
    return contact_matrix, contact_area_matrix

def save_results(contact_matrix, contact_area_matrix, output_file='particle_contacts.csv'):
    """接触分析結果をCSVファイルに保存"""
    num_particles = contact_matrix.shape[0] - 1
    results = []
    
    for i in range(1, num_particles + 1):
        # 各粒子の接触数を計算
        contacts = np.sum(contact_matrix[i, :] > 0)
        # 接触面積の合計を計算
        total_contact_area = np.sum(contact_area_matrix[i, :])
        
        # 接触している粒子のIDと接触面積を取得
        contacting_particles = []
        for j in range(1, num_particles + 1):
            if contact_matrix[i, j] > 0:
                contacting_particles.append(f"{j}({contact_area_matrix[i, j]:.1f}px²)")
        
        results.append({
            'Particle_ID': i,
            'Number_of_Contacts': contacts,
            'Total_Contact_Area': f"{total_contact_area:.1f}px²",
            'Contacting_Particles': ','.join(contacting_particles)
        })
    
    # DataFrameに変換して保存
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    logging.info(f"結果を保存しました: {output_file}")

def main():
    try:
        # マスクディレクトリの設定
        mask_dir = 'data/test/predicted_masks'
        
        # マスクスタックの読み込み
        logging.info("マスクスタックを読み込んでいます...")
        mask_stack = load_mask_stack(mask_dir)
        
        # 粒子のラベル付け
        logging.info("粒子をラベル付けしています...")
        labeled_stack, num_particles = label_particles(mask_stack)
        
        # 接触の分析
        logging.info("粒子間の接触を分析しています...")
        contact_matrix, contact_area_matrix = analyze_contacts(labeled_stack)
        
        # 結果の保存
        logging.info("結果を保存しています...")
        save_results(contact_matrix, contact_area_matrix)
        
        logging.info("分析が完了しました")
        
    except Exception as e:
        logging.error(f"分析中にエラーが発生しました: {str(e)}")
        raise

if __name__ == '__main__':
    main() 