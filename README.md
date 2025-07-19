# 🚀 Complete GitHub + Railway Deployment Guide

## 📁 **BƯỚC 1: Chuẩn bị Project Structure**

Đảm bảo project folder có cấu trúc như sau:

```
vietnamese-food-api/
├── app.py                           # Main API file
├── requirements.txt                 # Dependencies  
├── gunicorn_config.py              # Production server config
├── .env                            # Environment variables
├── .gitignore                      # Git ignore file
├── README.md                       # Documentation
├── Dockerfile                      # Docker config (optional)
└── models/                         # Model files
    ├── mobilenet_food_classifier.h5
    └── vietnamese_food_class_names.pkl
```

## 📝 **BƯỚC 2: Tạo file .gitignore**

Tạo file `.gitignore` để không upload files không cần thiết:

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment variables
.env
.venv
env/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
.DS_Store

# Logs
*.log
logs/

# Models (nếu quá lớn)
# models/*.h5  # Uncomment nếu model quá lớn (>100MB)
# models/*.pkl

# Temporary files
*.tmp
*.temp
.cache/

# Operating System
Thumbs.db
.DS_Store
```

## 📖 **BƯỚC 3: Tạo README.md**

```markdown
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
```

## 🍜 Supported Vietnamese Foods

Phở, Bánh Mì, Bún Bò Huế, Cơm Tấm, Bánh Xèo, Gỏi Cuốn, và 24 món khác...

## 🔧 Local Development

```bash
pip install -r requirements.txt
python app.py
```

## 📱 Mobile Integration

Compatible với Expo React Native - xem ScanFoodScreen.tsx để integrate.

## 📄 License

MIT License
```

## 🐙 **BƯỚC 4: Upload lên GitHub**

### **4.1: Khởi tạo Git repository**

```bash
# Mở Terminal/Command Prompt trong folder project
cd vietnamese-food-api

# Khởi tạo git
git init

# Add tất cả files
git add .

# Commit đầu tiên
git commit -m "🍜 Initial commit: Vietnamese Food Classifier API"
```

### **4.2: Tạo repository trên GitHub**

1. **Vào GitHub.com** và đăng nhập
2. **Click nút "+" (New repository)**
3. **Repository name**: `vietnamese-food-classifier-api`
4. **Description**: `🍜 AI-powered Vietnamese Food Recognition API`
5. **Public/Private**: Chọn Public
6. **✅ KHÔNG** check "Add README" (vì đã có rồi)
7. **Click "Create repository"**

### **4.3: Connect local với GitHub**

```bash
# Add remote origin (thay YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/vietnamese-food-classifier-api.git

# Đổi tên branch chính (nếu cần)
git branch -M main

# Push lên GitHub
git push -u origin main
```

**✅ Project đã được upload lên GitHub!**

## 🚂 **BƯỚC 5: Deploy lên Railway**

### **5.1: Đăng ký Railway**

1. **Vào [railway.app](https://railway.app)**
2. **Click "Login"**
3. **Sign up with GitHub** (recommended)
4. **Authorize Railway** để access GitHub repos

### **5.2: Deploy từ GitHub**

1. **Click "New Project"**
2. **Click "Deploy from GitHub repo"**
3. **Select repository**: `vietnamese-food-classifier-api`
4. **Click "Deploy Now"**

### **5.3: Configure Environment Variables**

Railway sẽ tự động:
- ✅ Detect Python project
- ✅ Install dependencies từ `requirements.txt`
- ✅ Run với Gunicorn

**Nếu cần custom settings:**

1. **Vào Project Dashboard**
2. **Click "Variables" tab**
3. **Add variables:**
   ```
   MODEL_PATH=models/mobilenet_food_classifier.h5
   CLASS_NAMES_PATH=models/vietnamese_food_class_names.pkl
   PORT=8000
   ```

### **5.4: Đợi Deployment**

- ⏱️ **Deploy time**: 5-10 phút
- 📊 **Logs**: Xem realtime logs trong Railway dashboard
- 🔗 **URL**: Railway sẽ cung cấp public URL

## 🔧 **BƯỚC 6: Test Deployment**

### **6.1: Get Railway URL**

1. **Trong Railway dashboard**
2. **Click "Settings" tab**
3. **Scroll xuống "Domains"**
4. **Copy URL**: `https://your-app-name.railway.app`

### **6.2: Test API**

```bash
# Test health check
curl https://your-app-name.railway.app/

# Expected response:
{
  "status": "healthy",
  "message": "🍜 Vietnamese Food Classifier API",
  "model_loaded": true,
  "classes_count": 30
}
```

### **6.3: Test với Postman**

1. **Open Postman**
2. **POST**: `https://your-app-name.railway.app/predict`
3. **Headers**: `Content-Type: application/json`
4. **Body (JSON)**:
```json
{
  "image": "your_base64_image_string_here",
  "top_k": 5
}
```

## 📱 **BƯỚC 7: Update Expo App**

Trong file ScanFoodScreen.tsx, thay đổi URL:

```typescript
// Thay đổi từ local
// const VIETNAMESE_FOOD_API_URL = 'http://127.0.0.1:5000';

// Thành Railway URL
const VIETNAMESE_FOOD_API_URL = 'https://your-app-name.railway.app';
```

## 🔄 **BƯỚC 8: CI/CD Auto Deploy**

Railway tự động deploy khi có changes trên GitHub:

```bash
# Make changes to code
git add .
git commit -m "🔧 Update API response format"
git push origin main

# Railway sẽ tự động detect và deploy lại!
```

## ⚠️ **TROUBLESHOOTING**

### **Model quá lớn (>500MB)**

```bash
# Option 1: Git LFS
git lfs track "*.h5"
git add .gitattributes
git add models/
git commit -m "🗃️ Add model with LFS"

# Option 2: Upload manual và download trong code
# Trong app.py, thêm function download model từ cloud storage
```

### **Railway Build Failed**

**Check logs để xem lỗi:**
1. Vào Railway dashboard
2. Click "Deployments" tab  
3. Click failed deployment
4. Xem logs để debug

**Common issues:**
- ❌ **Requirements.txt sai**: Check dependencies
- ❌ **Model files missing**: Ensure models/ folder uploaded
- ❌ **Memory limit**: Model quá lớn cho free tier

### **API Not Responding**

```bash
# Check Railway logs
# Trong dashboard: Deployments > Latest > View Logs

# Common fixes:
# 1. Ensure PORT environment variable set
# 2. Check gunicorn_config.py settings
# 3. Verify model files present
```

## 💰 **Railway Pricing**

- **Starter Plan**: $5/month
- **500 hours execution time**
- **8GB RAM**, **8 vCPU**
- **100GB storage**

**Free tier có limited resources - model lớn có thể cần upgrade.**

## 🎉 **COMPLETED DEPLOYMENT**

Sau khi hoàn thành:

1. ✅ **GitHub Repository**: Code được version control
2. ✅ **Railway Deployment**: API chạy trên cloud
3. ✅ **Auto CI/CD**: Git push → Auto deploy
4. ✅ **Public URL**: Accessible từ Expo app
5. ✅ **Scalable**: Railway auto-scales theo traffic

**🍜 Vietnamese Food Classifier API đã sẵn sàng phục vụ users trên toàn thế giới!**

---

## 📞 **Support**

Nếu gặp vấn đề:
1. Check Railway logs
2. Test locally trước
3. Verify GitHub sync
4. Check environment variables

**Happy coding! 🚀**