import os
import rasterio
import numpy as np
import rasterio.errors
from rasterio.windows import Window
import glob
from define_band import Sentinel2L2AData, Sentinel1GRDData
from tqdm import tqdm
class NDVIPreprocessing:
    def read_band(self, file_path, band_number):
        try:
            with rasterio.open(file_path) as src:
                band = src.read(band_number)
                profile = src.profile
            return band, profile
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening file {file_path}: {e}")
            return None, None

    def ndvi_raster_img(self, file_path, red_num, nir_num):
        red_band, _ = self.read_band(file_path, red_num)
        nir_band, _ = self.read_band(file_path, nir_num)
        if red_band is None or nir_band is None:
            return None
        ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
        return ndvi

    def create_dataset(self, directory):
        files = sorted(os.listdir(directory))
        ndvi_list = []
        for file in files:
            file_path = os.path.join(directory, file)
            print(f"Processing file: {file_path}")  # In ra đường dẫn tệp
            ndvi_raster = self.ndvi_raster_img(file_path, 8, 4)
            if ndvi_raster is not None:
                ndvi_list.append(ndvi_raster)
            else:
                print(f"Skipping file due to read error: {file_path}")
        return ndvi_list

    def make_ndvi_raster(self, file_path):
        try:
            with rasterio.open(file_path) as src:
                cloudy_prob = src.read(Sentinel2L2AData.CLD)  # Assuming this is a 2D array
                red_band, _ = self.read_band(file_path, Sentinel2L2AData.B04)
                nir_band, _ = self.read_band(file_path, Sentinel2L2AData.B08)
                
                if red_band is None or nir_band is None:
                    return None
                
                # Initialize ndvi_raster with the same shape as cloudy_prob
                ndvi_raster = np.zeros(cloudy_prob.shape)
                for x in range(cloudy_prob.shape[0]):
                    for y in range(cloudy_prob.shape[1]):
                        if cloudy_prob[x, y] < 35:  # Assuming this is the correct condition
                            ndvi_raster[x, y] = (nir_band[x, y] - red_band[x, y]) / (nir_band[x, y] + red_band[x, y] + 1e-10)
                        else:
                            ndvi_raster[x, y] = None
            return ndvi_raster
                    
        except rasterio.errors.RasterioIOError as e:
            print(f"Error opening file {file_path}: {e}")
            return None
    
    def make_S1_raster(self, file_path):
        try:
            with rasterio.open(file_path) as src:
                VV_band = src.read(Sentinel1GRDData.VV)
                VH_band = src.read(Sentinel1GRDData.VH)
            return VV_band, VH_band
        except rasterio.errors.RasterioIOError as e: 
            print(f"Eroor openning file")  
            
    def process_cloud_img(self, directory):
        ndvi_raster_list = []
        
        for file_name in tqdm(os.listdir(directory), desc='Processing: ---- Making Raster NDVI Image with cloud prob least ----'):
            file_path = f'{directory}/{file_name}'
            ndvi_raster = self.make_ndvi_raster(file_path)
            ndvi_raster_list.append(ndvi_raster)
        
        return ndvi_raster_list
class SplitTile():
    def __init__(self) -> None:
        pass

    def add_padding(self, image, tile_width, tile_height):
        # Tính toán kích thước mới
        h, w = image.shape[1], image.shape[2]
        new_h = (h + tile_height - 1) // tile_height * tile_height
        new_w = (w + tile_width - 1) // tile_width * tile_width

        # Tạo ảnh mới với padding
        padded_image = np.zeros((image.shape[0], new_h, new_w), dtype=image.dtype)
        padded_image[:, :h, :w] = image

        return padded_image

    def save_tile(self, src, tile, x, y, tile_width, tile_height, output_dir):
        window = Window(x, y, tile_width, tile_height)
        transform = src.window_transform(window)
        profile = src.profile
        profile.update({
            'height': tile_height,
            'width': tile_width,
            'transform': transform,
            'dtype': src.dtypes[0]  # Giữ nguyên kiểu dữ liệu
        })

        output_path = os.path.join(output_dir, f'tile_{x}_{y}.tif')
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(tile)

    def split_raster_to_tiles(self, raster_path, tile_width, tile_height, output_dir):
        with rasterio.open(raster_path) as src:
            image = src.read()
            padded_image = self.add_padding(image, tile_width, tile_height)

            n_cols, n_rows = padded_image.shape[2], padded_image.shape[1]

            for y in range(0, n_rows, tile_height):
                for x in range(0, n_cols, tile_width):
                    window = Window(x, y, tile_width, tile_height)
                    tile = padded_image[:, y:y + tile_height, x:x + tile_width]
                    self.save_tile(src, tile, x, y, tile_width, tile_height, output_dir)
    
    def pipeline(self, folder):
        # Lấy danh sách các tệp .tif trong thư mục
        image_paths = glob.glob(os.path.join(folder, "*.tif"))
        img_index = 0
        for img in image_paths:
            img_index += 1
            print(f"Processing image: {img}")

            # Tạo thư mục output tương ứng cho mỗi ảnh raster
            output_dir = os.path.join("assets/output_tiles", os.path.splitext(os.path.basename(img))[0])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            self.split_raster_to_tiles(img, tile_width=64, tile_height=64, output_dir=output_dir)

# if __name__ == '__main__':
#     test = NDVIPreprocessing()
#     ROOT_FOLDER = 'D:/Streamlit/api/assets/img'
#     #ndvi_raster = []
#     for folder_child in os.listdir(ROOT_FOLDER):
#         directory = f'{ROOT_FOLDER}/{folder_child}/S2L2A'
#         print(directory)
#         ndvi_raster_area = test.process_cloud_img(directory)
#         np.save(f'D:/Streamlit/api/assets/np/{folder_child}.npy',ndvi_raster_area)
        #ndvi_raster.append(ndvi_raster_area)

    #print(len(ndvi_raster))