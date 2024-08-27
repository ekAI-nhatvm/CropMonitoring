import rasterio
import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import pandas as pd
import os 
import datetime

class ExtractImgSat():
    def __init__(self, config):
        self.config = config

    # Đọc băng tần từ tệp TIFF
    def read_band(self, file_path, band_number):
        with rasterio.open(file_path) as src:
            band_data = src.read(band_number)
            profile = src.profile
        return band_data, profile

    # NDVI
    def calculate_ndvi(self, band1, band2):
        # Chuyển đổi kiểu dữ liệu của band1 và band2 sang float
        band1 = band1.astype(float)
        band2 = band2.astype(float)
        
        # Đảm bảo không có chia cho 0
        band_sum = band1 + band2
        band_sum[band_sum == 0] = np.nan  # Thay thế giá trị bằng NaN nếu tổng bằng 0
        ndvi = (band1 - band2) / band_sum
        ndvi[np.isnan(ndvi)] = 0  # Thay đổi NaN thành 0 nếu có
        return ndvi
    
    def calculate_evi(self, nir, red, blue, G=2.5, C1=6, C2=7.5, L=1):
        denominator = nir + C1 * red - C2 * blue + L
        
        # Thay thế giá trị 0 trong mẫu số bằng một giá trị nhỏ hơn
        denominator[denominator == 0] = 1e-10
        
        evi = G * (nir - red) / denominator
        evi[np.isnan(evi)] = 0 
        return evi

    def describes(self, v_index):
        sum_ndvi_diff_zero = 0
        cnt = 0
        
        # Tính tổng và đếm số lượng giá trị khác 0
        for i in range(0, v_index.shape[0]):
            for j in range(0, v_index.shape[1]):
                if v_index[i, j] != 0:
                    sum_ndvi_diff_zero += v_index[i, j]
                    cnt += 1
        
        # Kiểm tra cnt để tránh lỗi chia cho 0
        if cnt == 0:
            mean = float('nan')  # Hoặc chọn giá trị hợp lý khác
            print("Warning: cnt is zero, mean set to NaN")
        else:
            mean = float(sum_ndvi_diff_zero / cnt)
        
        # Tính giá trị max và min
        max_val = v_index.max()
        
        # Thay thế giá trị 0 bằng np.inf để tìm min
        modified_array = np.where(v_index == 0.0, np.inf, v_index)
        min_val = np.min(modified_array)

        return mean, max_val, min_val

    def pipeline_1_band(self, file_path, band_number1, band_number2):
        band1, _ = self.read_band(file_path, band_number1)
        band2, _ = self.read_band(file_path, band_number2)
        task = self.config['name_task']
        res_img = self.calculate_ndvi(band1,band2)
        

        mean, max, min = self.describes(res_img)
        return res_img, [file_path, mean, max, min]
    
    def pipeline_2_band(self, file_path, band_number1, band_number2, band_number3):
        nir, _ = self.read_band(file_path, band_number1)
        red, _ = self.read_band(file_path, band_number2)
        blue, _ = self.read_band(file_path, band_number3)
        res_img = self.calculate_evi(nir,red,blue)
        
        mean, max, min = self.describes(res_img)
        return res_img, [file_path, mean, max, min]
    
    def get_bands_for_index(self, name, indices):
        for index in indices:
            if index['name'] == name:
                return index.get('bands', {})
        return None

    def export_csv_file(self, vi_config):
        src = self.config['src']
        task = self.config['name_task']
        field = self.config['field']
        address = self.config['address']
        list_info = []
        indices = vi_config['indices']
        bands = self.get_bands_for_index(task, indices)
        if bands is None: 
            print('Khong co band nao duoc truy xuat')
        else:
            print(f'Caculating {task}')
            source_path = f'{src}{field}/'
            for file in os.listdir(source_path):
                if task == 'NDVI':
                    file_path = f'{source_path}{file}'
                    _, info = self.pipeline_1_band(file_path, bands['nir'], bands['red'])
                elif task == 'EVI':
                    file_path = f'{source_path}{file}'
                    _, info = self.pipeline_2_band(file_path, bands['nir'], bands['red'], bands['blue'])

                list_info.append({
                    'file_path':info[0],
                    'mean': info[1],
                    'max': info[2],
                    'min': info[3]
                })
            res = pd.DataFrame(list_info)
            res.to_csv(f'res/result_{task}_statistic_{address}_{field}.csv')
            print(f'Created res/result_{task}_statistic_{address}_{field}.csv')
   
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    print(f"Loaded config from {yaml_path}")
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Process some points with Sentinel Hub.')
    parser.add_argument('--config', required=True, help='Path to the YAML config file')
    # parser.add_argument('--visual', required=False, help='Visualization ?')
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    vi_config = load_yaml_config('cfg/vegetation_indices.yaml')

    process = ExtractImgSat(config)
    time_start  = datetime.start = datetime.datetime.now()
    process.export_csv_file(vi_config)
    time_end = datetime.datetime.now()
    print(f"All processing took {time_end - time_start}")

if __name__ == "__main__":
    main()