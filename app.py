import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import streamlit as st
import google.generativeai as genai
import base64
import io

# Load model CNN (DenseNet-121)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 2)  # 2 kelas: edible & poisonous
model.load_state_dict(torch.load("S:\College\mushroom_project\mushroom_classifier.pth", map_location=device))  # Load model yang sudah dilatih
model.to(device)
model.eval()

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Klasifikasi edible/poisonous
def predict_edibility(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(output, dim=1)[0]
        class_index = torch.argmax(probabilities).item()
    labels = ["Edible 🍄", "Poisonous ☠️"]
    return labels[class_index], probabilities[class_index].item()

# Masukkan API Key dari Google AI Studio
GOOGLE_API_KEY = "AIzaSyCAZli7ScTr3YbFamxol3dM-KY2nCx86fM"
genai.configure(api_key=GOOGLE_API_KEY)

def get_mushroom_info(image):
    # Cek format gambar
    print(f"Format gambar: {image.format}")  # Debugging untuk cek format gambar

    # Konversi gambar ke Base64
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)  # Simpan dengan format aslinya
    img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

    # Sesuaikan MIME type berdasarkan format gambar
    mime_type = f"image/{image.format.lower()}"  # Contoh: image/jpeg atau image/png

    # Buat model
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Kirim request ke API
    response = model.generate_content([
        "Tolong identifikasi genus dan spesies jamur dalam gambar ini.",
        {
            "mime_type": mime_type,
            "data": img_base64
        }
    ])

    return response.text  # Ambil hasil deskripsi dari API

# Streamlit UI
st.title("🍄 Identifikasi Jamur Beracun dan Layak Dikonsumsi")
st.write("Aplikasi berbasis AI untuk pengenalan jamur makroskopis secara otomatis")

uploaded_file = st.file_uploader("Unggah gambar jamur", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("Cek sekarang 🚀"):
        with st.spinner("Menganalisis gambar..."):
            # Prediksi edible/poisonous
            edibility, confidence = predict_edibility(image)
            
            # Kirim ke OpenAI untuk info tambahan
            mushroom_info = get_mushroom_info(image)

            # Tampilkan hasil
            st.subheader("📌 Hasil:")
            st.write(f"**{edibility}** (Tingkat kepercayaan: {confidence:.2%})")
            
            st.subheader("🔍 Informasi Tambahan:")
            st.write(mushroom_info)