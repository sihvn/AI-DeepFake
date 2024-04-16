import math
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from torchvision import models, transforms

# from process_data import get_predict_inputs


def predict(
    video_path: str,
    model: models.ResNet | models.EfficientNet,
    device: torch.device,
    num_frames=5,
):
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = math.floor(total_frames // (num_frames + 1))

    if interval == 0:
        interval = 1

    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame.thumbnail((200, 200), Image.Resampling.LANCZOS)
            frames.append(frame)

    cap.release()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model.eval()

    inputs = torch.stack([transform(frame) for frame in frames])
    inputs = inputs.to(device)
    outputs = model(inputs)
    _, preds = torch.max(outputs, 1)
    result = "Fake" if torch.mean(preds.float()) > 0.4 else "Real"

    return preds, result, frames
