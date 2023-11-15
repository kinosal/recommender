"""Run the program."""

import rekognition as rek
import gpt

if __name__ == "__main__":
    image_file_names = ["nik1.jpeg", "nik2.jpeg", "nik3.jpeg"]
    topic = "travel destinations"

    labels = set()
    for image_file_name in image_file_names:
        image_name = rek.hash_and_scale_image(mode="name", image_name=image_file_name)
        if not rek.find_image(image_name):
            image_url = rek.upload_image(image_name)
        else:
            image_url = f"https://{rek.BUCKET}.s3.amazonaws.com/{image_name}"
        # objects = rek.detect_labels(image_name)  # Use AWS Rekognition
        objects = gpt.detect_labels(image_url)  # Use OpenAI
        labels.update(objects)
        print(objects)

    recommendations = gpt.recommend(labels, topic)
    print(recommendations)
