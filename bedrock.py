"""API calls to Amazon Bedrock GenAI models."""

import json
import boto3


def recommend(labels: list, topic: str, model: str):
    """Create recommendations based on a topic and labels."""
    recommender = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
    body = json.dumps(
        {
            "prompt": (
                "System Instruction: You recommend the user anything they could do, based on the topic they are interested in and the provided list of labels recognized in their personal photos. Your up to five recommendations are concise, specific and tailored to the personal user preferences you can infer from the available information. You answer with a numbered list, with each recommendation on a new line. You do not repeat the instructions or provide any additional content.\n\n"
                f"Prompt: Here is the list of labels from my personal photos: {', '.join(labels)}. Given this information, which {topic} would you recommend to me?\n\n"
                "Response:\n"
            ),
            "max_gen_len": 256,
            "temperature": 0.5,
            "top_p": 1,
        }
    )
    modelId = "meta.llama2-13b-chat-v1"
    accept = "application/json"
    contentType = "application/json"
    response = recommender.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("generation").strip()
