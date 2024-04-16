import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog

import torch
from PIL import ImageTk

from model import *
from predict import *


class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Deepfake Detection")
        self.root.resizable(False, False)

        self.frame1 = ttk.Frame(self.root, width=600)
        self.frame1.grid(row=0, padx=10, pady=5, sticky="ew")
        self.root.grid_columnconfigure(0, weight=1)

        self.btn_load_video = tk.Button(
            self.frame1, text="Select a Video", command=self.select_video
        )
        self.btn_load_video.grid(row=0, column=0, pady=20, padx=20)

        self.video_path_label = tk.Label(self.frame1, text="")
        self.video_path_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))

        self.frame2 = ttk.Frame(self.root)
        self.frame2.grid(row=1, padx=10, pady=5, sticky="ew")
        self.root.grid_columnconfigure(1, weight=1)

        self.images = []
        self.image_text = []

        self.frame3 = ttk.Frame(self.root)
        self.frame3.grid(row=2, padx=10, pady=5, sticky="ew")
        self.root.grid_columnconfigure(2, weight=1)

        self.result_label = tk.Label(self.frame3, text="")
        self.result_label.grid(row=0, column=0, pady=20)
        self.root.mainloop()

    def set_image_panels(self, num_frames=5):
        self.images = []

        for i in range(num_frames):
            panel = tk.Label(self.frame2)
            panel.grid(row=0, column=i, padx=10, pady=10)
            # panel.grid(row=1, column=i, padx=10, pady=10)
            self.images.append(panel)

        for i in range(num_frames):
            self.frame2.grid_columnconfigure(i, weight=1)

    def set_image_text_panels(self, num_frames):
        self.image_text = []

        for i in range(num_frames):
            panel = tk.Label(self.frame2)
            panel.grid(row=1, column=i, padx=10, pady=10)
            self.image_text.append(panel)

        for i in range(num_frames):
            self.frame2.grid_columnconfigure(i, weight=1)

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        device = self.get_device()
        model = get_model("ResNet50", device)
        model_weights_path = "weights/ResNet50_Adam_LR1e-3_WD1e-5_DP0_EP15.pth"
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        return model, device

    def display_frames(self, frames, preds):
        self.set_image_panels(len(frames))

        for i in range(len(frames)):
            img = ImageTk.PhotoImage(frames[i])
            self.images[i].config(image=img)
            self.images[i].image = img

        self.set_image_text_panels(len(preds))

        for i in range(len(preds)):
            if preds[i] == 0:
                prediction_label = "Real"
            else:
                prediction_label = "Fake"

            self.image_text[i].config(text=prediction_label)

    def select_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.video_path_label.config(text=file_path)

            model, device = self.load_model()

            preds, result, _, frames = predict(file_path, model, device)

            self.display_frames(frames, preds)
            self.result_label.config(text=f"Result: {result}")


if __name__ == "__main__":
    GUI()
