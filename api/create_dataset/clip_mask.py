import rasterio
from rasterio.mask import mask
import fiona
import os
from tqdm import tqdm
import argparse
import yaml

class ClipMaskByPixel():
    def __init__(self, config) -> None:
        self.config = config

        self.src_dir = self.config['src']
        self.dst_dir = self.config['dst']
        self.field_index = self.config['field_index']
        self.name = self.config['name']
        self.shapefile_path = f'{self.config['shp']}{self.name}{self.field_index}/{self.name}{self.field_index}.shp'
        

        # Tạo thư mục đích nếu chưa tồn tại
        self.dst_dir_field = f'{self.dst_dir}{self.name}{self.field_index}'
        if not os.path.exists(self.dst_dir_field):
            os.makedirs(self.dst_dir_field)

    def get_list_dir_src(self):
        img_tif_list = []
        for file in os.listdir(self.src_dir):
            img_tif_list.append(f'{self.src_dir}/{file}')
        return img_tif_list

    # Function to read shapes from a shapefile
    def read_shapes(self):
        shape_path = f'{self.shapefile_path}'
        with fiona.open(shape_path, "r") as shapefile:
            shapes = [feature["geometry"] for feature in shapefile]
        return shapes

    # Function to clip a raster by shapes
    def clip_raster_by_shape(self, raster_path, shapes):
        #path_file = f'{self.src_dir}{self.field_index}/{raster_path}'
        path_file = f'{self.src_dir}/{raster_path}'
        with rasterio.open(path_file) as src:
            out_image, out_transform = mask(src, shapes, crop=True)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Tạo đường dẫn đầy đủ cho tệp TIFF mới
            output_file = f'{self.dst_dir_field}/mask_{os.path.basename(raster_path)}'
            
            with rasterio.open(output_file, "w", **out_meta) as dest:
                dest.write(out_image)

    def process(self):
        shapes = self.read_shapes()
        #for raster_path in tqdm(os.listdir(f'{self.src_dir}{self.field_index}')):
        for raster_path in tqdm(os.listdir(f'{self.src_dir}')):
            self.clip_raster_by_shape(raster_path, shapes)

def load_yaml_config(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    print(f"Loaded config from {yaml_path}")
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Clipping mask raster file')
    parser.add_argument('--config', required=True, help='Path to the YAML config file')
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    clip = ClipMaskByPixel(config)
    clip.process()

if __name__ == "__main__":
    main()