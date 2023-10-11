from typing import Dict

import gradio as gr
import requests

# model, labels = load_model()  # Use the function from app.py


def classify_image(filepath: str) -> Dict[str, float]:
    with open(filepath, "rb") as f:
        # Make the API request
        response = requests.post("http://18.225.9.52/predict", files={"file": f})
    predictions = response.json()["predictions"]
    print(predictions)
    print(list(predictions.keys())[0])
    print(list(predictions.values())[0])
    return predictions


gr.Interface(
    fn=classify_image,
    inputs=gr.Image(
        shape=(224, 224),
        source="upload",
        label="Upload Image for Classification",
        type="filepath",
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicted Class"),
    live=False,
).launch()
