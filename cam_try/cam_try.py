import torch
import cv2
import numpy as np
from torchvision import transforms

# Download model MiDaS dari PyTorch Hub
model = torch.load('C:/Users/ideap/MiDaS')

# Set model ke eval mode
model.eval()

# Gambar input
img = cv2.imread("path_to_image.jpg")
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_input = transform(img).unsqueeze(0)

# Prediksi kedalaman
with torch.no_grad():
    depth_map = model(img_input)

# Konversi ke depth map yang bisa dipakai untuk visualisasi
depth_map = depth_map.squeeze().cpu().numpy()
depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
depth_map = np.uint8(depth_map)

# Tampilkan depth map menggunakan OpenCV atau Matplotlib
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
