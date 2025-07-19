# ==========================================
# VIETNAMESE FOOD CLASSIFIER API - TF 2.18.0 Compatible
# ==========================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image
import io
import base64
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
model = None
class_names = None

# Vietnamese food names mapping
VIETNAMESE_NAMES = {
    'banh_beo': 'Bánh Bèo',
    'banh_bot_loc': 'Bánh Bột Lọc', 
    'banh_can': 'Bánh Căn',
    'banh_canh': 'Bánh Canh',
    'banh_chung': 'Bánh Chưng',
    'banh_cuon': 'Bánh Cuốn',
    'banh_duc': 'Bánh Đúc',
    'banh_gio': 'Bánh Giò',
    'banh_khot': 'Bánh Khọt',
    'banh_mi': 'Bánh Mì',
    'banh_pia': 'Bánh Pía',
    'banh_tet': 'Bánh Tét',
    'banh_trang_nuong': 'Bánh Tráng Nướng',
    'banh_xeo': 'Bánh Xèo',
    'bun_bo_hue': 'Bún Bò Huế',
    'bun_dau_mam_tom': 'Bún Đậu Mắm Tôm',
    'bun_mam': 'Bún Mắm',
    'bun_rieu': 'Bún Riêu',
    'bun_thit_nuong': 'Bún Thịt Nướng',
    'ca_kho_to': 'Cá Kho Tộ',
    'canh_chua': 'Canh Chua',
    'cao_lau': 'Cao Lầu',
    'chao_long': 'Cháo Lòng',
    'com_tam': 'Cơm Tấm',
    'goi_cuon': 'Gỏi Cuốn',
    'hu_tieu': 'Hủ Tiếu',
    'mi_quang': 'Mì Quảng',
    'nem_chua': 'Nem Chua',
    'pho': 'Phở',
    'xoi_xeo': 'Xôi Xéo'
}

# ==========================================
# TF 2.18.0 + KERAS 3.x COMPATIBLE MODEL LOADING
# ==========================================

def load_model_and_classes():
    """Load model compatible with TF 2.18.0 + Keras 3.x"""
    global model, class_names
    
    try:
        logger.info("🔄 Loading model...")
        logger.info(f"🔧 TensorFlow version: {tf.__version__}")
        
        model_path = os.getenv('MODEL_PATH', 'models/mobilenet_food_classifier.h5')
        class_names_path = os.getenv('CLASS_NAMES_PATH', 'models/vietnamese_food_class_names.pkl')
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model with TF 2.18.0 compatibility
        logger.info(f"📂 Loading model from: {model_path}")
        
        # For TF 2.18.0, try direct loading first
        try:
            model = tf.keras.models.load_model(model_path)
            logger.info("✅ Direct model loading successful!")
        except Exception as e:
            logger.warning(f"⚠️ Direct loading failed: {str(e)}")
            
            # Try loading without compilation
            logger.info("🔧 Trying to load without compilation...")
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Recompile with current TF version
            logger.info("🔧 Recompiling model...")
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            logger.info("✅ Model recompiled successfully!")
        
        # Load class names
        if os.path.exists(class_names_path):
            with open(class_names_path, 'rb') as f:
                class_names = pickle.load(f)
            logger.info(f"✅ Class names loaded: {len(class_names)} classes")
        else:
            # Fallback to default class names
            class_names = list(VIETNAMESE_NAMES.keys())
            logger.warning("⚠️ Class names file not found, using default names")
        
        # Verify model
        logger.info(f"🎯 Model loaded successfully!")
        logger.info(f"   📐 Input shape: {model.input_shape}")
        logger.info(f"   📐 Output shape: {model.output_shape}")
        logger.info(f"   🏷️  Classes: {len(class_names)}")
        
        # Test prediction to ensure everything works
        logger.info("🧪 Testing model with dummy input...")
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        test_pred = model.predict(dummy_input, verbose=0)
        logger.info(f"✅ Model test successful! Output shape: {test_pred.shape}")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise e

# ==========================================
# IMAGE PROCESSING
# ==========================================

def preprocess_image(image_data, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        raise e

# ==========================================
# PREDICTION
# ==========================================

def predict_food(image_array, top_k=5):
    """Make prediction on preprocessed image"""
    try:
        # Make prediction
        predictions = model.predict(image_array, verbose=0)
        
        # Handle different output shapes from Keras 3.x
        if len(predictions.shape) > 1:
            predictions = predictions[0]  # Remove batch dimension
        
        # Get top-k predictions
        top_indices = np.argsort(predictions)[::-1][:top_k]
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(class_names):  # Safety check
                class_name = class_names[idx]
                vietnamese_name = VIETNAMESE_NAMES.get(class_name, class_name.replace('_', ' ').title())
                confidence = float(predictions[idx])
                
                results.append({
                    'rank': i + 1,
                    'class_id': class_name,
                    'class_name': vietnamese_name,
                    'confidence': confidence,
                    'percentage': round(confidence * 100, 2)
                })
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error making prediction: {str(e)}")
        raise e

# ==========================================
# API ROUTES
# ==========================================

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': '🍜 Vietnamese Food Classifier API',
        'model_loaded': model is not None,
        'classes_count': len(class_names) if class_names else 0,
        'tensorflow_version': tf.__version__,
        'keras_version': tf.keras.__version__ if hasattr(tf.keras, '__version__') else 'N/A'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Server is still loading the model. Please try again.'
            }), 503
        
        # Get request data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image provided',
                'message': 'Please provide image data in base64 format'
            }), 400
        
        # Preprocess image
        logger.info("🖼️ Processing image...")
        image_array = preprocess_image(data['image'])
        
        # Make prediction
        logger.info("🤖 Making prediction...")
        top_k = data.get('top_k', 5)
        predictions = predict_food(image_array, top_k)
        
        if not predictions:
            return jsonify({
                'error': 'No predictions generated',
                'message': 'Model could not generate predictions for this image'
            }), 500
        
        # Get best prediction
        best_prediction = predictions[0]
        
        logger.info(f"✅ Prediction completed: {best_prediction['class_name']} ({best_prediction['percentage']:.1f}%)")
        
        return jsonify({
            'success': True,
            'best_prediction': {
                'food_name': best_prediction['class_name'],
                'confidence': best_prediction['confidence'],
                'percentage': best_prediction['percentage']
            },
            'top_predictions': predictions,
            'model_info': {
                'total_classes': len(class_names),
                'model_type': 'MobileNetV2 Transfer Learning',
                'tensorflow_version': tf.__version__,
                'keras_version': tf.keras.__version__ if hasattr(tf.keras, '__version__') else 'N/A'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all available food classes"""
    try:
        if class_names is None:
            return jsonify({
                'error': 'Classes not loaded'
            }), 503
        
        classes_info = []
        for i, class_name in enumerate(class_names):
            vietnamese_name = VIETNAMESE_NAMES.get(class_name, class_name.replace('_', ' ').title())
            classes_info.append({
                'id': i,
                'class_id': class_name,
                'vietnamese_name': vietnamese_name
            })
        
        return jsonify({
            'total_classes': len(class_names),
            'classes': classes_info
        })
        
    except Exception as e:
        logger.error(f"❌ Error getting classes: {str(e)}")
        return jsonify({
            'error': 'Failed to get classes',
            'message': str(e)
        }), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction for multiple images"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        images = data.get('images', [])
        
        if not images:
            return jsonify({'error': 'No images provided'}), 400
        
        if len(images) > 10:  # Limit batch size
            return jsonify({'error': 'Too many images. Maximum 10 per batch.'}), 400
        
        results = []
        
        for i, image_data in enumerate(images):
            try:
                image_array = preprocess_image(image_data)
                predictions = predict_food(image_array, top_k=3)
                
                results.append({
                    'image_index': i,
                    'success': True,
                    'predictions': predictions
                })
                
            except Exception as e:
                results.append({
                    'image_index': i,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total_images': len(images),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {str(e)}")
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e)
        }), 500

# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on the server'
    }), 500

# ==========================================
# STARTUP
# ==========================================

def create_app():
    """Application factory"""
    
    # Load model on startup
    try:
        load_model_and_classes()
        logger.info("🚀 API ready to serve requests!")
    except Exception as e:
        logger.error(f"❌ Failed to start API: {str(e)}")
        # Continue without model for health checks
    
    return app

# ==========================================
# MAIN
# ==========================================

if __name__ == '__main__':
    # Development server
    app = create_app()
    
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('DEBUG', 'False').lower() == 'true'
    
    logger.info(f"🌟 Starting development server on port {port}")
    logger.info(f"🔧 TensorFlow version: {tf.__version__}")
    app.run(host='0.0.0.0', port=port, debug=debug)