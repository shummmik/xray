import cv2
import numpy as np
import streamlit as st
import torch
from model import Unet

SHAPE_DOWN = (256, 256)
FLAG = cv2.IMREAD_GRAYSCALE
BODY = "Pulmonary Chest X-Ray Defect Detection"
WAIT = "Wait for the model to process the image"

st.title(BODY, anchor=None)

input_file = st.file_uploader("Upload a PNG File", type=["png"])
if (input_file is not None) and input_file.name.endswith(".png"):
    nparr = np.frombuffer(input_file.getvalue(), np.uint8)
    img_np = cv2.imdecode(nparr, FLAG)
    img = cv2.resize(img_np, SHAPE_DOWN)
    st.image(img)

    wait = st.subheader(WAIT, anchor=None)

    np_img = np.array(img)
    np_img = np_img / 255

    model = torch.load("model.pt")

    device = torch.device("cpu")

    data = torch.tensor(np_img.reshape(1, 1, 256, 256)).to(device=device).float()
    model = model.to(device=device)

    out = model(data)
    out = torch.round(out).reshape(256, 256).detach().numpy()
    st.image(out)
