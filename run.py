"""Run the program."""

import rekognition as rek
import gpt

if __name__ == "__main__":
    image_file_names = ["nik1.jpeg", "nik2.jpeg"]
    topic = "books"
    model = "Amazon Rekognition"

    labels = set()
    for image_file_name in image_file_names:
        hashed_image, hashed_image_name = rek.hash_and_scale_image(
            mode="path", image_name=image_file_name
        )
        if not rek.find_image(hashed_image_name):
            image_url = rek.upload_image(mode="path", image_name=hashed_image_name)
        else:
            image_url = f"https://{rek.BUCKET}.s3.amazonaws.com/{hashed_image_name}"
        if model == "Amazon Rekognition":
            objects = rek.detect_labels(hashed_image_name)
        elif model == "GPT-4 Vision":
            objects = gpt.detect_labels(image_url)
        labels.update(objects)
        print(objects)

    recommendations = gpt.recommend(labels, topic)
    print(recommendations)
