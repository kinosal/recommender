"""Functions for image recognition with AWS Rekognize."""

from hashlib import sha256
from PIL import Image
import boto3

BUCKET = "recommender-images"


def hash_and_scale_image(
        mode: str, image_name: str = None, image_file=None
    ) -> str:
    """Hash image with sha256."""
    if mode == "name":
        with open(f"images/{image_name}", "rb") as image:
            image_content = image.read()
    elif mode == "file":
        image_name = image_file.name
        image_content = image_file.read()

    suffix = image_name.split(".")[-1]
    hashed_name = sha256(image_content).hexdigest() + "." + suffix
    image = Image.open(image_file)
    image.thumbnail((1024, 1024))
    image.save(f"images/{hashed_name}")

    return hashed_name


def find_image(image_name: str) -> bool:
    """Check if image exists in S3."""
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(BUCKET)
    for obj in bucket.objects.all():
        if obj.key == image_name:
            return True
    return False


def upload_image(image_name: str) -> str:
    """Upload image from images folder to S3."""
    uploader = boto3.client(service_name="s3")
    uploader.upload_file(Filename=f"images/{image_name}", Bucket=BUCKET, Key=image_name)
    return f"https://{BUCKET}.s3.amazonaws.com/{image_name}"


def detect_labels(
        image_name: str, min_confidence: int = 90, max_labels: int = 50
    ) -> list:
    """Detect labels in image."""
    recognizer = boto3.client(service_name="rekognition")
    response = recognizer.detect_labels(
        Image={"S3Object": {"Bucket": BUCKET, "Name": image_name}},
        Features=["GENERAL_LABELS"],
        MinConfidence=min_confidence,
        MaxLabels=max_labels,
    )
    return [label["Name"] for label in response["Labels"]]
