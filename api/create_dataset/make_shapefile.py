import geopandas as gpd
from shapely.geometry import Polygon

# Danh sách các tọa độ của các polygon (mỗi polygon được định nghĩa bởi một danh sách các tọa độ điểm)
# Mỗi polygon được định nghĩa bởi một danh sách các điểm, với mỗi điểm là một tuple hoặc danh sách có hai giá trị (x, y)
polygons = [
    [(108.121626, 13.907033), (108.123772, 13.905867), (108.121744, 13.902024), (108.119706, 13.903024), (108.121626, 13.907033)]
]

# Tạo các đối tượng Polygon từ tọa độ
shapes = [Polygon(coords) for coords in polygons]

# Tạo GeoDataFrame từ các đối tượng Polygon
gdf = gpd.GeoDataFrame(geometry=shapes)

# Xác định hệ tọa độ (CRS) nếu cần. Ví dụ: EPSG:4326 cho WGS84
gdf.set_crs(epsg=4326, inplace=True)

# Xuất ra shapefile
shapefile_path = "hungson.shp"
gdf.to_file(shapefile_path)

print(f"Shapefile đã được tạo tại: {shapefile_path}")

