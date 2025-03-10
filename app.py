import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import streamlit as st
import google.generativeai as genai
import base64
import io
import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# **1. Load CNN Model for Edibility Classification**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_edibility = models.densenet121(pretrained=False)
num_features = model_edibility.classifier.in_features
model_edibility.classifier = nn.Linear(num_features, 2)  # 2 classes: edible & poisonous
try:
    model_edibility.load_state_dict(torch.load("models/edible_mushroom_classifier.pth", map_location=device))
except Exception as e:
    st.error(f"Error loading edible classifier model: {e}")
model_edibility.to(device)
model_edibility.eval()

# **2. Load Model for "Mushroom or Not" Detection**
model_mushroom = models.densenet121(pretrained=False)
num_features = model_mushroom.classifier.in_features
model_mushroom.classifier = nn.Linear(num_features, 2)  # 2 classes: mushroom & non-mushroom
try:
    model_mushroom.load_state_dict(torch.load("models/nonmushroom_classifier.pth", map_location=device))
except Exception as e:
    st.error(f"Error loading non-mushroom classifier model: {e}")
model_mushroom.to(device)
model_mushroom.eval()

# **3. Image Transformation**
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# **4. Function to Detect "Is This a Mushroom?"**
def is_mushroom(image):
    image = image.convert("RGB")  # To ensure the image has 3 channels (RGB)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_mushroom(image)
        probabilities = torch.softmax(output, dim=1)[0]
        class_index = torch.argmax(probabilities).item()
    
    labels = ["Jamur ✅", "Non-Jamur ❌"]
    return labels[class_index], probabilities[class_index].item()

# **5. Function to Predict Edibility**
def predict_edibility(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model_edibility(image)
        probabilities = torch.softmax(output, dim=1)[0]
        class_index = torch.argmax(probabilities).item()
    
    labels = ["Layak dikonsumsi 🍄", "Beracun ☠️"]
    return labels[class_index], probabilities[class_index].item()

# **6. Configure Google Gemini API**
genai.configure(api_key=GOOGLE_API_KEY)

# **7. Function to Get Mushroom Information from Google Gemini**
def get_mushroom_info(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content([
        "Tolong identifikasi genus dan spesies jamur dalam gambar ini.",
        {"mime_type": "image/jpeg", "data": img_base64}
        ])
        if response and hasattr(response, 'text'):
            return response.text
        else:
            return "Informasi tidak tersedia."
    except Exception as e:
        return f"Terjadi kesalahan saat mengambil informasi: {e}"

# **8. Streamlit UI**

# ================== Custom Styling ==================
st.markdown(
    """
    <style>
        /* Font styling */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Header Title */
        .title {
            font-size: 2.5rem;  /* Ukuran font lebih besar agar lebih proporsional */
            font-weight: 700;  /* Lebih tebal agar terlihat lebih kuat */
            color: #ff4b4b;  /* Warna tetap merah */
            text-align: left;  /* Judul diratakan ke kiri */
            margin-bottom: 15px;  
            line-height: 1.3;  /* Jarak antar baris lebih proporsional */
        }

        /* Box untuk Catatan */
        .warning-box {
            background-color: #2b1e1e;  /* Warna latar belakang lebih gelap */
            color: #ff6b6b;  /* Warna teks merah agar lebih mencolok */
            border-left: 5px solid #ff4b4b;  /* Garis tepi di sisi kiri */
            padding: 12px;
            margin: 15px 0;
            border-radius: 8px;
            font-weight: bold;
            font-size: 1rem;
            text-align: center;
        }

        /* Upload Box */
        div[data-testid="stFileUploader"] {
            border: 2px dashed #4CAF50;
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        div[data-testid="stFileUploader"]:hover {
            border: 2px solid #2E7D32;
        }

        /* Radio Button */
        div[data-testid="stRadio"] > label {
            font-weight: bold;
            font-size: 1.1rem;
        }

        /* Animasi Hover */
        div[data-testid="stButton"] > button:hover {
            background-color: #4CAF50 !important;
            color: white !important;
            transform: scale(1.05);
            transition: all 0.3s ease-in-out;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# ================== Header with Mushroom Icon ==================
st.markdown(
    """
    <h1 class="title">🍄 Selamat Datang di Aplikasi Deteksi Jamur</h1>
    """,
    unsafe_allow_html=True
)

st.write(
    "Aplikasi ini menggunakan teknologi **Deep Learning** untuk mengidentifikasi apakah suatu gambar adalah "
    "**jamur atau bukan**. Jika terdeteksi sebagai jamur, aplikasi juga akan menentukan apakah jamur tersebut "
    "**layak dikonsumsi atau beracun**."
)

# Warning Box for User
st.markdown(
    '<div class="warning-box">⚠️ <b>Peringatan:</b> Hasil analisis ini hanya berbasis gambar dan AI. <br> Tetap waspada dan jangan konsumsi jamur tanpa konfirmasi dari ahli mikologi!</div>',
    unsafe_allow_html=True
)

st.write("Unggah gambar jamur untuk mengetahui apakah jamur tersebut **beracun atau layak dikonsumsi**.")

# Image input method selection
option = st.radio("Pilih metode input gambar:", ["📂 Unggah dari file", "📸 Ambil foto langsung"])

uploaded_file = None
camera_file = None

if option == "📂 Unggah dari file":
    uploaded_file = st.file_uploader("Unggah gambar jamur", type=["jpg", "jpeg", "png"])
elif option == "📸 Ambil foto langsung":
    camera_file = st.camera_input("Ambil foto langsung 📸")

# **Use the selected image**
if uploaded_file or camera_file:
    image = Image.open(uploaded_file if uploaded_file else camera_file)

    # **Check if the image is a mushroom**
    label_mushroom, confidence_mushroom = is_mushroom(image)
    
    if label_mushroom == "Jamur ✅":
        st.image(image, caption="✅ Gambar terdeteksi sebagai jamur.", use_container_width=True)

        if st.button("Cek sekarang 🚀"):
            with st.spinner("Menganalisis gambar..."):
                # Predict edible/poisonous
                edibility, confidence_edibility = predict_edibility(image)

                # Fetch additional information using Google Gemini
                mushroom_info = get_mushroom_info(image)

                # **Display results**
                st.subheader("📌 Hasil:")
                st.write(f"**{edibility}** (Tingkat kepercayaan: {confidence_edibility:.2%})")

                st.subheader("🔍 Informasi Tambahan:")
                st.write(mushroom_info)
    else:
        st.warning("⚠️ Aplikasi tidak mendeteksi jamur. Mohon unggah gambar jamur yang lebih jelas.")
