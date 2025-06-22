import streamlit as st
import os
import numpy as np
from PIL import Image
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device), processor, device

model, processor, device = load_model()

def prepare_dataset(folder="custom_images", count=20):
    os.makedirs(folder, exist_ok=True)
    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    dataset = CIFAR10(root="data", download=True, transform=transform)

    for i in range(count):
        img, label = dataset[i]
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(folder, f"{i}_{dataset.classes[label]}.jpg"))

    return folder

@st.cache_resource
def build_index(image_folder):
    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    embeddings = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_emb = model.get_image_features(**inputs).cpu().numpy()
        embeddings.append(image_emb)

    embedding_array = np.vstack(embeddings).astype('float32')
    faiss.normalize_L2(embedding_array)
    index = faiss.IndexFlatIP(embedding_array.shape[1])
    index.add(embedding_array)

    return index, image_paths

def search(query, mode="text", top_k=5):
    if mode == "text":
        inputs = processor(text=query, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            emb = model.get_text_features(**inputs).cpu().numpy()
    else:
        inputs = processor(images=query, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs).cpu().numpy()

    faiss.normalize_L2(emb)
    _, indices = index.search(emb.astype('float32'), top_k)
    return [image_paths[i] for i in indices[0]]

st.set_page_config(page_title="VLM Search Engine", layout="wide")
st.title("🔍 Visual-Language Multimodal Search Engine")

with st.spinner("Preparing dataset..."):
    image_dir = prepare_dataset()
    index, image_paths = build_index(image_dir)
I
mode = st.radio("Choose Search Mode:", ["Text Search", "Image Search"])

if mode == "Text Search":
    query = st.text_input("Enter a description (e.g., 'car', 'frog'):")
    if st.button("Search"):
        results = search(query, mode="text")
        st.subheader("Results:")
        st.image(results, width=200)

else:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", width=200)
        if st.button("Search"):
            results = search(image, mode="image")
            st.subheader("Results:")
            st.image(results, width=200)
