# 🍄 Edible vs Poisonous Mushroom Classification

## 📌 Project Overview
This project is a deep learning-based web application that classifies mushrooms as edible or poisonous using Convolutional Neural Networks (CNN). The model is built with PyTorch and deployed using Streamlit. Additionally, Google's Gemini AI model, Gemini-2.0-Flash, provides extra information about the detected mushrooms.

## 🚀 Features
✅ Detects if an image contains a mushroom or not

✅ Classifies mushrooms as edible or poisonous

✅ Provides additional genus and species information using Google Gemini AI

✅ Simple web interface with Streamlit

## 📂 Folder Structure
```
mushroom_project/
├── models/                      	# Trained models (not included in GitHub)
│   ├── edible_mushroom_classifier.pth
│   ├── nonmushroom_classifier.pth
│
├── dataset/                      	# Dataset (not included in GitHub)
│   ├── dataset_mushroom_edibility/
│   │   ├── edible_mushroom/
│   │   ├── poisonous_mushroom/
│   │
│   ├── dataset_mushroom_vs_nonmushroom/
│       ├── mushroom/
│       ├── nonmushroom/
│
├── app.py                        	# Streamlit web app
├── mushroom_preprocessing.ipynb  	# Preprocessing edible/poisonous dataset
├── nonmushroom_preprocessing.ipynb	# Preprocessing mushroom/non-mushroom dataset
├── .env.example                   	# API key configuration (rename to .env)
├── requirements.txt               	# Dependencies
├── README.md                      	# Project documentation
```

## 📊 Dataset
This project uses publicly available datasets for training and evaluation:

- **[Mushroom Edibility Dataset](https://www.kaggle.com/datasets/marcosvolpato/edible-and-poisonous-fungi)**  
  Contains images of edible and poisonous mushrooms used for classification.  
- **[Non-Mushroom Dataset](https://www.kaggle.com/datasets/shamsaddin97/image-captioning-dataset-random-images)**  
  Used for distinguishing between mushroom and non-mushroom images. Since we already have mushroom images, we use this dataset to classify mushroom vs non-mushroom. 

## 🔧 Installation & Setup
1️⃣ Clone this repository:
```
git clone https://github.com/syahelrusfi21/Edible-vs-Poisonous-Mushroom-Classification.git
cd Edible-vs-Poisonous-Mushroom-Classification
```
2️⃣ Create a virtual environment (optional but recommended):
```
python -m venv env
source env/bin/activate  # On macOS/Linux
env\Scripts\activate     # On Windows
```
3️⃣ Install dependencies:
```
pip install -r requirements.txt
```
4️⃣ Set up API key for Google Gemini AI:
```
Rename .env.example to .env
Add your GOOGLE_API_KEY inside .env
```
5️⃣ Run the application:
```
streamlit run app.py
```

## 📊 Model Training
The CNN model is trained using DenseNet121 as the base model.
Two models were trained:
- Mushroom vs. Non-Mushroom Classifier
- Edible vs. Poisonous Mushroom Classifier

## ⚠️ Disclaimer
🚨 This application is for educational purposes only. Do NOT consume mushrooms based solely on AI predictions. Always consult with a professional mycologist before consuming any wild mushrooms.

## 🌟 Acknowledgments
- PyTorch for deep learning framework
- Streamlit for web app
- Google Gemini API for additional mushroom information
