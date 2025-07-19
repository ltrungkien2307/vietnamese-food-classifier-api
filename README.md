# 🍜 Vietnamese Food Classifier API

AI-powered API để nhận diện 30 món ăn Việt Nam sử dụng TensorFlow và MobileNetV2.

## 🚀 Features

- 🤖 Nhận diện 30 món ăn Việt Nam
- 📊 Trả về top 5 predictions với confidence scores
- 🔥 TensorFlow 2.18.0 + MobileNetV2
- ⚡ Flask API với CORS support
- 📱 Tương thích với Expo React Native

## 🏗️ Tech Stack

- **Backend**: Python Flask
- **AI Model**: TensorFlow 2.18.0, MobileNetV2
- **Server**: Gunicorn
- **Deployment**: Railway

## 📋 API Endpoints

- `GET /` - Health check
- `POST /predict` - Phân tích ảnh món ăn
- `GET /classes` - Danh sách 30 món ăn
- `POST /batch_predict` - Phân tích nhiều ảnh

## 🧪 Usage

```bash
# Health check
curl https://your-api.railway.app/

# Predict food
curl -X POST https://your-api.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "base64_image_string"}'