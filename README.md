# GPT Recommendations from Images

Live version of this app at https://recommendation.streamlit.app

## Description

This [Streamlit](https://streamlit.io) mini-app recommends any activity by analyzing a few personal photos. It uses Amazon [Rekognition](https://aws.amazon.com/rekognition/image-features/) or OpenAI's [GPT Vision](https://platform.openai.com/docs/models) to detect people, objects, places, and sentiment in the uploaded images, and OpenAI's [GPT Text](https://platform.openai.com/docs/models) to generate respective recommendations.

The Rekognition API does not require a prompt and is configured to return up to 50 labels with at least 90% confidence for each image. For image recognition with OpenAI's GPTs, the app generates a prompt to return labels as a comma-separated list. The app then uses another OpenAI GPT model to generate personal recommendations for the specified topic based on the image labels.

## Contribution

I hope this will be of value to people seeking to understand GPTs' current capabilities and the overall progress of natural language processing (NLP) and generative AI. Plus maybe you can even use the app to generate some recommendations you might find interesting.

Please reach out to me at nikolas@schriefer.me with any feedback - especially suggestions to improve this - or questions you might have.
