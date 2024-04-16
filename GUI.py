import tkinter as tk
from tkinter import filedialog
import tkinter.ttk as ttk
from PIL import Image, ImageTk
import cv2
import torch
from torchvision import models
import torch.nn as nn
import math
from torchvision import transforms

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Deepfake Detection")
        self.root.resizable(False, False)

        self.frame1 = ttk.Frame(self.root, width=600)
        self.frame1.grid(row=0, padx=10, pady=5, sticky='ew')
        self.root.grid_columnconfigure(0, weight=1) 

        self.btn_load_video = tk.Button(self.frame1, text="Select a Video", command=self.select_video)
        self.btn_load_video.grid(row=0, column=0, pady=20, padx=20)

        self.video_path_label = tk.Label(self.frame1, text="")
        self.video_path_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))

        self.frame2 = ttk.Frame(self.root)
        self.frame2.grid(row=1, padx=10, pady=5, sticky='ew')
        self.root.grid_columnconfigure(1, weight=1)  


        self.images = []
        for i in range(4):
            panel = tk.Label(self.frame2)
            panel.grid(row=0, column=i, padx=10, pady=10)
            self.images.append(panel)


        for i in range(4):
            self.frame2.grid_columnconfigure(i, weight=1)

        self.frame3 = ttk.Frame(self.root)
        self.frame3.grid(row=2, padx=10, pady=5, sticky='ew')
        self.root.grid_columnconfigure(2, weight=1)  

        self.result_label = tk.Label(self.frame3, text="")
        self.result_label.grid(row=0, column=0, pady=20)
        self.root.mainloop()

    def get_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        device = self.get_device()
        model = models.resnet152(pretrained=False)  
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  
        

        model.load_state_dict(torch.load('weights/ResNet152_Adam_LR1e-3_WD0_DP0_EP15.pth', map_location=device))
        model.to(device)
        model.eval()
        return model, device

    def predict(self, frames, model, device):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        inputs = torch.stack([transform(frame) for frame in frames])
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        result = 'Fake' if torch.mean(preds.float()) > 0.3 else 'Real'
        self.result_label.config(text=f"Result: {result}")

    def display_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // 4

        frames = []
        for i in range(4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
            ret, frame = cap.read()
            if ret:
                frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frame.thumbnail((200, 200), Image.Resampling.LANCZOS) 
                img = ImageTk.PhotoImage(frame)
                self.images[i].config(image=img)
                self.images[i].image = img
                frames.append(frame)

        cap.release()
        model, device = self.load_model()
        self.predict(frames, model, device)

    def select_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.video_path_label.config(text=file_path) 
            self.display_frames(file_path)

if __name__ == "__main__":
    GUI()
