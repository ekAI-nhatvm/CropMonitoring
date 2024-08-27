from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    read_data,
)
import datetime
import os
import math
import argparse
import pandas as pd
import yaml
from tqdm import tqdm
from shapely.geometry import MultiLineString, Polygon, box, shape
import rasterio
from rasterio.transform import from_bounds


class SentinelHubProcessorPoint:
    def __init__(self, config, name_collection):
        self.sh_config = self.init_sh_config(config["credentials"])
        self.output_folder = config["output_folder"]
        self.geo_json = config["json_polygon"]
        self.resolution = config["resolution"]
        self.start_date = config["start_date"]
        self.end_date = config["end_date"]
        self.limit_of_api_request = (
            2500 * self.resolution
        )  # meters (limit of image size is equal to 2500px)
        self.name_collection = name_collection

    def init_sh_config(self, credentials):
        sh_config = SHConfig()
        sh_config.sh_client_id = credentials["sh_client_id"]
        sh_config.sh_client_secret = credentials["sh_client_secret"]
        sh_config.sh_base_url = credentials["sh_base_url"]
        sh_config.sh_auth_base_url = credentials["sh_auth_base_url"]
        return sh_config

    def create_bbox(self):
        geo_json = read_data(self.geo_json)
        area = shape(geo_json)
        bbox = BBox(area, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=self.resolution)

        return bbox, size

    @staticmethod
    def meters_to_degrees(
        meters, latitude
    ):  # kinh tuyến đường dọc, vĩ tuyến đường ngang
        lat_deg = meters / 111320  # 1 kinh độ khoảng 111320m
        lon_deg = meters / (111320 * math.cos(math.radians(latitude)))
        return lat_deg, lon_deg

    @staticmethod
    def create_evalscript():
        return """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12", "CLD", "SCL"],
                    units: "DN"
                }],
                output: {
                    bands: 14,
                    sampleType: "INT16"
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B01,
                    sample.B02,
                    sample.B03,
                    sample.B04,
                    sample.B05,
                    sample.B06,
                    sample.B07,
                    sample.B08,
                    sample.B8A,
                    sample.B09,
                    sample.B11,
                    sample.B12,
                    sample.CLD,
                    sample.SCL];
        }
        """

    def get_true_color_request(self, time_interval, bbox_coordinates, bbox_size):
        evalscript_true_color = self.create_evalscript()
        return SentinelHubRequest(
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                    # mosaicking_order=MosaickingOrder.LEAST_CC
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox_coordinates,
            size=bbox_size,
            config=self.sh_config,
        )

    @staticmethod
    def create_evalscript_sentinel1():
        return """
        //VERSION=3
        function setup() {
            return {
                input: ["VH", "VV"],
                output: { id:"default", bands: 2}
            };
        }

        function evaluatePixel(samples) {
            return [toDb(samples.VH), toDb(samples.VV)]
        }

        // visualizes decibels from -20 to +10
        function toDb(linear) {
            var log = 10 * Math.log(linear) / Math.LN10
            return Math.max(0, (log + 20) / 30)
        }
        """

    def get_data_sentinel1_grd(self, time_interval, bbox_coordinates, bbox_size):
        evalscript = self.create_evalscript_sentinel1()
        return SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_IW,
                    time_interval=time_interval,
                    mosaicking_order=MosaickingOrder.MOST_RECENT,
                    #other_args={"processing": {"orthorectify": "True"}}
                    other_args={"processing": {
                            "backCoeff": "SIGMA0_ELLIPSOID",  # Radiometric calibration
                            "orthorectify": "True",             # Terrain correction
                            "speckleFilter": {"type": "LEE","windowSizeX": 5,"windowSizeY": 5}, # speckle filter
                            "demInstance": "COPERNICUS" # DEM 
                        }
                    }
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.TIFF)
            ],
            bbox=bbox_coordinates,
            size=bbox_size,
            config=self.sh_config
        )

    def download_and_save_images(self, requests, slots, bbox):
        data = SentinelHubDownloadClient(config=self.sh_config).download(
            requests, max_threads=5
        )
        folder_name = f"{self.output_folder}/{self.name_collection}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for idx, image in tqdm(
            enumerate(data), total=len(requests), desc="Downloading Images"
        ):
            start_date, end_date = slots[idx]
            print(
                f"Downloading image {idx + 1}/{len(requests)} for time interval {start_date} to {end_date}"
            )

            image_filename = f"{folder_name}/{start_date}_{end_date}.tif"

            width, height = image.shape[1], image.shape[0]
            transform = from_bounds(*bbox, width, height)

            metadata = {
                "driver": "GTiff",
                "height": height,
                "width": width,
                "count": image.shape[2],
                "dtype": image.dtype,
                "crs": "EPSG:4326",  # Assuming the bbox is in EPSG:4326
                "transform": transform,
            }

            with rasterio.open(image_filename, "w", **metadata) as dst:
                for band in range(image.shape[2]):
                    dst.write(image[:, :, band], band + 1)

            print(f"Saved {image_filename}")

    def process(self, points, numbers):
        for point_center, number in zip(points, numbers):
            time_start = datetime.datetime.now()

            point_folder = os.path.join(self.output_folder, str(number))
            bbox_cal = self.calculate_bbox_coordinate(point_center)
            bbox_coordinates = BBox(bbox=bbox_cal, crs=CRS.WGS84)
            bbox_size = bbox_to_dimensions(bbox_coordinates, resolution=self.resolution)
            print("Bbox Coordinate: ", bbox_coordinates)
            print("Boxsize: ", bbox_size)

            start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
            end = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")

            # Generate weekly intervals
            weekly_intervals = pd.date_range(
                start=start, end=end, freq="W-WED"
            )  # Change 'W-WED' to 'W-SUN', 'W-MON', etc., if you prefer a different day

            # Convert intervals to list of tuples (start, end)
            intervals = [
                (
                    str(weekly_intervals[i].date().isoformat()),
                    str(weekly_intervals[i + 1].date().isoformat()),
                )
                for i in range(len(weekly_intervals) - 1)
            ]

            # Include the last interval to end date
            if weekly_intervals[-1] < pd.Timestamp(end):
                intervals.append(
                    (
                        str(weekly_intervals[-1].date().isoformat()),
                        str(end.date().isoformat()),
                    )
                )

            print(f"\nProcessing point {number}: {point_center}")

            # check parse collection "S2L2A or S1GRD
            list_of_requests = []
            for interval in tqdm(
                intervals, desc=f"Creating requests for point {number}", unit="request"
            ):
                slot_start, slot_end = interval
                if self.name_collection == "S2L2A":
                    request = self.get_true_color_request(
                        (slot_start, slot_end), bbox_coordinates, bbox_size
                    )
                elif self.name_collection == "S1GRD":
                    request = self.get_data_sentinel1_grd(
                        (slot_start, slot_end), bbox_coordinates, bbox_size
                    )

                list_of_requests.append(request.download_list[0])

            self.download_and_save_images(
                list_of_requests, intervals, point_folder, bbox_coordinates
            )

            time_end = datetime.datetime.now()
            print(
                f"Process and Download time for point {number} is {time_end - time_start}"
            )

    def process_single(self):
        time_start = datetime.datetime.now()
        bbox_coordinates, bbox_size = self.create_bbox()

        print(f"-----bbox coor: {bbox_coordinates}, bbox size: {bbox_size}-----")
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")

        # Generate weekly intervals
        weekly_intervals = pd.date_range(
            start=start, end=end, freq="W-WED"
        )  # Change 'W-WED' to 'W-SUN', 'W-MON', etc., if you prefer a different day

        # Convert intervals to list of tuples (start, end)
        intervals = [
            (
                str(weekly_intervals[i].date().isoformat()),
                str(weekly_intervals[i + 1].date().isoformat()),
            )
            for i in range(len(weekly_intervals) - 1)
        ]

        # Include the last interval to end date
        if weekly_intervals[-1] < pd.Timestamp(end):
            intervals.append(
                (
                    str(weekly_intervals[-1].date().isoformat()),
                    str(end.date().isoformat()),
                )
            )

        list_of_requests = []
        for interval in tqdm(intervals, desc=f"Creating requests", unit="request"):
            slot_start, slot_end = interval

            if self.name_collection == "S2L2A":
                request = self.get_true_color_request(
                    (slot_start, slot_end), bbox_coordinates, bbox_size
                )
            elif self.name_collection == "S1GRD":
                request = self.get_data_sentinel1_grd(
                    (slot_start, slot_end), bbox_coordinates, bbox_size
                )

            list_of_requests.append(request.download_list[0])

        self.download_and_save_images(list_of_requests, intervals, bbox_coordinates)
        time_end = datetime.datetime.now()
        print(f"Process and Download time is: {time_end - time_start}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process some points with Sentinel Hub."
    )
    parser.add_argument("--config", required=True, help="Path to the YAML config file")
    parser.add_argument(
        "--collection", required=True, help="Collection name (S2L2A or S1GRD)"
    )
    return parser.parse_args()


def load_yaml_config(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    print(f"Loaded config from {yaml_path}")
    return config


def read_points_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    points = list(zip(df["longitude"], df["latitude"]))
    numbers = df["number"].tolist()
    print(f"Read {len(points)} points from {csv_path}")
    return points, numbers


def main():
    args = parse_args()
    config = load_yaml_config(args.config)
    name_collection = args.collection
    processor = SentinelHubProcessorPoint(config, name_collection)
    time_start = datetime.datetime.now()
    processor.process_single()
    time_end = datetime.datetime.now()
    print(f"All processing took {time_end - time_start}")


if __name__ == "__main__":
    main()
