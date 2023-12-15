import argparse
import glob
import traceback
import pandas as pd
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def blend_images(path, scale):
    img_files = [f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))]
    if len(img_files) < 2:
        return None
    img1 = cv2.imread(os.path.join(path, img_files[0]))
    img2 = cv2.imread(os.path.join(path, img_files[-1]))
    blended = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    return blended, scale.replace(",", "x").replace(" ", "").replace("(", "").replace(")", "")

def process_data(root, output, ratio, index):
    csv_files = glob.glob(os.path.join(root, '*.csv'))
    df_list = [pd.read_csv(f) for f in csv_files]
    data = pd.concat(df_list)

    assert index in ['psnr', 'ssim', 'lpips', 'chamfer']
    ascending = True if index in ['lpips', 'chamfer'] else False
    data.sort_values(by=[index, 'path'], ascending=ascending, inplace=True)
    data.drop_duplicates(subset='path', keep='first', inplace=True)

    train, test = train_test_split(data, train_size=ratio[0], random_state=42)
    val, test = train_test_split(test, train_size=ratio[1] / (ratio[1] + ratio[2]), random_state=42)

    for dataset, name in tqdm(zip([train, val, test], ['train', 'val', 'test'])):
        for _, row in tqdm(dataset.iterrows()):
            
            src_path = row['path']
            scale = row['scale'].replace(",", "x").replace(" ", "").replace("(", "").replace(")", "")
            link_name = f"{'_'.join(row['path'].split('/')[-2:])}"
            link_path_dir = os.path.join(output, name, scale)
            os.makedirs(link_path_dir, exist_ok=True)
            link_path = os.path.join(link_path_dir, link_name)
            
            # link src_path dir to link_path dir
            if not os.path.exists(link_path):
                try:
                    os.symlink(src_path, link_path)
                except:
                    print(f"Error creating symlink for {src_path} to {link_path}, {traceback.format_exc()}")
            
            # img, scale = blend_images(row['path'], row['scale'])
            # if img is not None:
            #     folder_path = os.path.join(output, name, scale)
            #     os.makedirs(folder_path, exist_ok=True)
            #     file_name = f"{'_'.join(row['path'].split('/')[-2:])}.jpg"
            #     cv2.imwrite(os.path.join(folder_path, file_name), img)

def main():
    parser = argparse.ArgumentParser(description='Process and organize images dataset.')
    parser.add_argument('--root', type=str, required=True, help='Root folder path containing CSV files')
    parser.add_argument('--output', type=str, required=True, help='Output folder path for the processed dataset')
    parser.add_argument('--index', type=str, required=True, help='Index to make the dataset')
    parser.add_argument('--ratio', type=float, nargs=3, help='Ratio for train, val, and test splits')
    args = parser.parse_args()

    process_data(args.root, args.output, args.ratio, args.index)

if __name__ == "__main__":
    main()
