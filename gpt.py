"""OpenAI GPT API calls."""

import os
import openai


openai.api_key = os.getenv("OPENAI_API_KEY")


def detect_labels(image_url: str) -> list:
    """Detect labels in an image."""
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": "You recognize people, objects, places, and sentiment in an image using computer vision. You extract and describe up to twenty detected labels precisely and accurately and output them as a comma-separated list.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "List the objects in this image:"},
                    {
                        "type": "image_url",
                        "image_url": image_url,
                    },
                ],
            },
        ],
        max_tokens=128,
    )
    return response.choices[0].message.content.split(", ")


def recommend(labels: list, topic: str, model: str) -> str:
    """Create recommendations based on a topic and preference labels."""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You recommend the user anything they could do, based on the topic they are interested in and the provided list of labels recognized in their personal photos. Your up to five recommendations are concise, specific and tailored to the personal user preferences you can infer from the available information. You answer in numbered list form, with each recommendation on a new line, without any additional content.",
            },
            {
                "role": "user",
                "content": (
                    f"Here is the list of labels from my personal photos: {labels}.\n"
                    f"Given this information, which {topic} would you recommend to me?"
                ),
            },
        ],
        temperature=0.7,
        max_tokens=256,
    )
    return response.choices[0].message.content
