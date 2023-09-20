from typing import Any, Dict, Union

import gradio as gr
import numpy as np
from PIL import Image

from app import load_model, predict, preprocess_image  # import necessary functions

model, labels = load_model()  # Use the function from app.py


def classify_image(inp: Union[np.ndarray, Any]) -> Dict[str, float]:
    image = Image.fromarray(np.uint8(inp)).convert("RGB")
    x = preprocess_image(image)  # Use the function from app.py
    predictions = predict(x)  # Use the function from app.py
    return predictions


gr.Interface(
    fn=classify_image,
    inputs=gr.Image(
        shape=(224, 224), source="upload", label="Upload Image for Classification"
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicted Class"),
    examples=["/Users/harrison/Desktop/msds/msds_fall23/MLops/apple_pie.jpg"],
    live=False,
).launch()
