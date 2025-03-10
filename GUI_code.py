import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from PIL import Image as PILImage
from fastai.vision import *
from fastai.vision.interpret import *
from fastai.callbacks.hooks import *
from pathlib import Path
from fastai.utils.mem import *
from fastai.vision import get_transforms
import os
import datetime
size = (228, 352)
free = gpu_mem_get_free_no_cache()
if free > 8200: bs=8
else:           bs=4
print(f"using bs={bs}, have {free}MB of GPU RAM free")
codes = np.loadtxt('data\\codes.txt', dtype=str)
path_img = "data\\image"
print(path_img)
path_lbl = "data\\labels"
print(path_lbl)
get_y_fn = lambda x: path_lbl+"/"+f'{x.stem}{x.suffix}'
src = (SegmentationItemList.from_folder(path_img)
       .split_by_fname_file('C:\\Users\\BHARGAV RAM\\Desktop\\roaddetection\\valid.txt')
       .label_from_func(get_y_fn, classes=codes))
print(src)
tfms = get_transforms(do_flip=True, flip_vert=False, max_rotate=0.0, max_zoom=1.0, max_lighting=0.1, max_warp=0.0)
data = (src.transform(tfms, size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

def acc_rtk(input, target):
    target = target.squeeze(1)
    mask = target != 0
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
# Load the U-Net with ResNet34 model
model = torch.load('C:\\Users\\BHARGAV RAM\\Desktop\\roaddetection\\finalroad.pkl', map_location=torch.device('cpu'))
if isinstance(model, dict):
    # If the model is saved as a dictionary, extract the 'model' key
    model = model['model']
model.eval()
# Transformation for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
def predict_image(image):
    # Preprocess the image
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        prediction = model(image)

    # Convert the prediction to a numpy array
    prediction = prediction.squeeze().numpy()

    return prediction

# Function to colorize the segmentation mask
def colorfull(prediction):
    prediction = np.argmax(prediction, axis=0)
    width, height = prediction.shape
    colors = np.array([
        (0, 0, 0),      # Background
        (85, 85, 255),   # Road Asphalt
        (85, 170, 127),  # Road Paved
        (255, 170, 127), # Road Unpaved
        (255, 255, 255), # Road Marking
        (255, 85, 255),  # Speed Bump
        (255, 255, 127), # Cats Eye
        (170, 0, 127),   # Storm Drain
        (0, 255, 255),   # Manhole Cover
        (0, 0, 127),     # Patches
        (170, 0, 0),     # Water Puddle
        (255, 0, 0),     # Pothole
        (255, 85, 0),    # Cracks
    ], dtype=np.uint8)
    colored_image = colors[prediction.astype(int)]
    return colored_image
# Streamlit app
def main():
    st.title("Road Surface Detection")
    uploaded_file = st.file_uploader("Choose an image to upload...", type=["jpg", "png", "jpeg"])
    start = datetime.datetime.now()
    if uploaded_file is not None:
        image = PILImage.open(uploaded_file)
        if image.mode == 'L':
            image = image.convert('RGB')
        st.image(image, caption="Uploaded Image.", use_column_width=True)
        if st.button("Predict"):
            st.text("Making prediction...")
            prediction = predict_image(image)
            end = datetime.datetime.now()
            colormap = plt.get_cmap("tab20")
            colored_pred = colormap(prediction.argmax(axis=0) / 13.0)[:, :, :3]
            st.image(colored_pred, caption="Segmentation Mask", use_column_width=True)
            colored_prediction = colorfull(prediction)
            st.image(PILImage.fromarray(colored_prediction), caption="Colored Segmentation Mask", use_column_width=True)
            st.write("TIme taken to predict :{}".format(end - start))
            legend = "C:\\Users\\BHARGAV RAM\\Desktop\\roaddetection\\Screenshot 2023-10-28 181342.png"
            leg_img = PIL.Image.open(legend)
            st.image(leg_img, caption="Legend",use_column_width=True)
if __name__ == "__main__":
    main()
