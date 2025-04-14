import os
import config
import pandas as pd

from io import BytesIO
from PIL import Image
from minio import Minio


class MinioUtils:
    _minio_client = None

    @staticmethod
    def get_minio_client():
        if MinioUtils._minio_client is None:
            try:
                MinioUtils._minio_client = Minio(
                    endpoint=os.getenv("MINIO_ENDPOINT"),
                    access_key=os.getenv("MINIO_ACCESS_KEY"),
                    secret_key=os.getenv("MINIO_SECRET_KEY"),
                    secure=False,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Minio client: {e}")

        return MinioUtils._minio_client

    @staticmethod
    def load_image_from_minio(object_path: str) -> Image.Image:
        client = MinioUtils.get_minio_client()
        response = client.get_object(os.getenv("MINIO_BUCKET"), object_path)
        return Image.open(BytesIO(response.read())).convert("RGB")

    @staticmethod
    def load_text_file_from_minio(object_path: str) -> str:
        client = MinioUtils.get_minio_client()
        response = client.get_object(os.getenv("MINIO_BUCKET"), object_path)
        return response.read().decode("utf-8")

    @staticmethod
    def load_csv_from_minio(object_path: str) -> pd.DataFrame:
        client = MinioUtils.get_minio_client()
        response = client.get_object(os.getenv("MINIO_BUCKET"), object_path)
        return pd.read_csv(BytesIO(response.read()))