
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


load_dotenv()


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
# IMPROVED MODEL LOADING - MATCH COLAB VERSION
# ==========================================

def load_model_and_classes():
    """Load model compatible with TF 2.18.0 and match Colab results"""
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
        
        try:
            # Try direct loading first (for TF 2.18.0)
            model = tf.keras.models.load_model(model_path)
            logger.info("✅ Direct model loading successful!")
            
        except Exception as e:
            logger.warning(f"⚠️ Direct loading failed: {str(e)}")
            logger.info("🔧 Trying alternative loading methods for TF 2.18.0...")
            
            # Method 1: Load without compilation
            try:
                model = tf.keras.models.load_model(model_path, compile=False)
                logger.info("✅ Loaded without compilation")
                
                # Recompile for TF 2.18.0
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                logger.info("✅ Model recompiled for TF 2.18.0")
                
            except Exception as e2:
                logger.warning(f"⚠️ Recompilation failed: {str(e2)}")
                
                # Method 2: Load with custom objects (if needed)
                try:
                    model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects=None,
                        compile=False,
                        safe_mode=False  # New in TF 2.16+
                    )
                    logger.info("✅ Loaded with safe_mode=False")
                    
                except Exception as e3:
                    logger.error(f"❌ All loading methods failed: {str(e3)}")
                    raise e3
        
        # Load class names in exact same order as Colab
        if os.path.exists(class_names_path):
            with open(class_names_path, 'rb') as f:
                class_names = pickle.load(f)
            logger.info(f"✅ Class names loaded: {len(class_names)} classes")
        else:
            # Use EXACT same class names as in Colab
            class_names = [
                'banh_beo', 'banh_bot_loc', 'banh_can', 'banh_canh', 'banh_chung',
                'banh_cuon', 'banh_duc', 'banh_gio', 'banh_khot', 'banh_mi',
                'banh_pia', 'banh_tet', 'banh_trang_nuong', 'banh_xeo', 'bun_bo_hue',
                'bun_dau_mam_tom', 'bun_mam', 'bun_rieu', 'bun_thit_nuong', 'ca_kho_to',
                'canh_chua', 'cao_lau', 'chao_long', 'com_tam', 'goi_cuon',
                'hu_tieu', 'mi_quang', 'nem_chua', 'pho', 'xoi_xeo'
            ]
            logger.warning("⚠️ Class names file not found, using Colab order")
        
        # Verify model matches Colab
        logger.info(f"🎯 Model verification:")
        logger.info(f"   📐 Input shape: {model.input_shape}")
        logger.info(f"   📐 Output shape: {model.output_shape}")
        logger.info(f"   🏷️  Classes: {len(class_names)}")
        
        # Verify class count matches model output
        expected_classes = model.output_shape[-1]
        if len(class_names) != expected_classes:
            logger.error(f"❌ Class mismatch! Model expects {expected_classes}, got {len(class_names)}")
            raise ValueError(f"Class count mismatch: model={expected_classes}, classes={len(class_names)}")
        
        # Test prediction with dummy input to ensure TF 2.18.0 compatibility
        logger.info("🧪 Testing model compatibility with TF 2.18.0...")
        dummy_input = tf.random.normal((1, 224, 224, 3), dtype=tf.float32)
        try:
            test_pred = model(dummy_input, training=False)  # Use call instead of predict for testing
            logger.info(f"✅ Model test successful! Output shape: {test_pred.shape}")
        except Exception as test_error:
            logger.warning(f"⚠️ Model test warning: {str(test_error)}")
            # Try with predict method
            test_pred = model.predict(dummy_input, verbose=0)
            logger.info(f"✅ Model predict test successful! Output shape: {test_pred.shape}")
        
        logger.info("✅ Model and classes verified successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        raise e

# ==========================================
# IMPROVED IMAGE PROCESSING - MATCH COLAB EXACTLY
# ==========================================

def preprocess_image_colab_style(image_data, img_size=224):
    """
    Preprocess image EXACTLY like in Colab for consistent results
    Matching: keras.preprocessing.image methods
    """
    try:
        # Decode base64 image
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
        
        # Load image exactly like Colab
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary (like Colab)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to target size (optimized for TF 2.18.0)
        image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)  # Better quality than NEAREST
        
        # Convert to array exactly like keras.preprocessing.image.img_to_array
        img_array = np.array(image, dtype=np.float32)
        
        # Add batch dimension like Colab: np.expand_dims(img_array, axis=0)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize EXACTLY like Colab: / 255.0
        img_array = img_array / 255.0
        
        # Ensure TF 2.18.0 compatibility
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        logger.info(f"🖼️ Image preprocessed: shape={img_array.shape}, dtype={img_array.dtype}")
        logger.info(f"   📊 Value range: [{tf.reduce_min(img_array):.3f}, {tf.reduce_max(img_array):.3f}]")
        
        return img_array
        
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        raise e

# ==========================================
# IMPROVED PREDICTION - MATCH COLAB EXACTLY
# ==========================================

def predict_food_colab_style(image_array, top_k=5):
    """
    Make prediction EXACTLY like in Colab with TF 2.18.0 compatibility
    """
    try:
        # Make prediction with TF 2.18.0 optimizations
        logger.info("🤖 Making prediction with TF 2.18.0...")
        
        # Method 1: Use model.predict (like Colab)
        try:
            predictions = model.predict(image_array, verbose=0)
            logger.info("✅ Using model.predict() method")
        except Exception as e:
            logger.warning(f"⚠️ model.predict() failed: {str(e)}")
            # Method 2: Use direct call
            predictions = model(image_array, training=False)
            predictions = predictions.numpy()  # Convert to numpy for TF 2.18.0
            logger.info("✅ Using model() call method")
        
        # Handle predictions exactly like Colab
        # In Colab: predictions[0] to get first (and only) prediction
        if len(predictions.shape) > 1:
            prediction_probs = predictions[0]  # Remove batch dimension
        else:
            prediction_probs = predictions
        
        # Convert to numpy if it's still a tensor (TF 2.18.0)
        if hasattr(prediction_probs, 'numpy'):
            prediction_probs = prediction_probs.numpy()
        
        # Get top class exactly like Colab: np.argmax(predictions[0])
        predicted_class_idx = np.argmax(prediction_probs)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction_probs[predicted_class_idx])
        
        # Get top predictions exactly like Colab: np.argsort(predictions[0])[::-1][:5]
        top_indices = np.argsort(prediction_probs)[::-1][:top_k]
        
        logger.info(f"🤖 Prediction results:")
        logger.info(f"   🏆 Top prediction: {predicted_class} ({confidence:.4f})")
        
        results = []
        for i, idx in enumerate(top_indices):
            if idx < len(class_names):  # Safety check
                class_name = class_names[idx]
                vietnamese_name = VIETNAMESE_NAMES.get(class_name, class_name.replace('_', ' ').title())
                prob = float(prediction_probs[idx])
                
                results.append({
                    'rank': i + 1,
                    'class_id': class_name,
                    'class_name': vietnamese_name,
                    'confidence': prob,
                    'percentage': round(prob * 100, 2)
                })
                
                logger.info(f"   #{i+1}: {class_name} -> {prob:.4f} ({prob*100:.1f}%)")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error making prediction: {str(e)}")
        raise e

# ==========================================
# COLAB-STYLE WRAPPER FUNCTION
# ==========================================

def predict_vietnamese_food_api(image_data, img_size=224, top_k=5):
    """
    Complete prediction pipeline matching Colab exactly
    This replicates the predict_vietnamese_food function from Colab
    """
    try:
        # Step 1: Preprocess image exactly like Colab
        img_array = preprocess_image_colab_style(image_data, img_size)
        
        # Step 2: Make prediction exactly like Colab  
        results = predict_food_colab_style(img_array, top_k)
        
        if not results:
            raise ValueError("No predictions generated")
        
        # Step 3: Return results in same format as Colab
        best_result = results[0]
        
        return {
            'predicted_class': best_result['class_id'],
            'confidence': best_result['confidence'],
            'top_classes': [r['class_id'] for r in results],
            'top_probs': [r['confidence'] for r in results],
            'all_results': results
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction pipeline error: {str(e)}")
        raise e

# ==========================================
# API ROUTES - UPDATED
# ==========================================

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': '🍜 Vietnamese Food Classifier API - Colab Compatible',
        'model_loaded': model is not None,
        'classes_count': len(class_names) if class_names else 0,
        'tensorflow_version': tf.__version__,
        'preprocessing': 'Colab-style'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - Colab compatible"""
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
        
        # Use Colab-style prediction
        logger.info("🖼️ Processing image with Colab-style preprocessing...")
        
        top_k = data.get('top_k', 5)
        img_size = data.get('img_size', 224)  # Allow custom image size
        
        prediction_result = predict_vietnamese_food_api(
            data['image'], 
            img_size=img_size, 
            top_k=top_k
        )
        
        logger.info(f"✅ Prediction completed: {prediction_result['predicted_class']} "
                   f"({prediction_result['confidence']:.4f})")
        
        return jsonify({
            'success': True,
            'best_prediction': {
                'food_name': VIETNAMESE_NAMES.get(
                    prediction_result['predicted_class'], 
                    prediction_result['predicted_class'].replace('_', ' ').title()
                ),
                'class_id': prediction_result['predicted_class'],
                'confidence': prediction_result['confidence'],
                'percentage': round(prediction_result['confidence'] * 100, 2)
            },
            'top_predictions': prediction_result['all_results'],
            'colab_format': {
                'predicted_class': prediction_result['predicted_class'],
                'confidence': prediction_result['confidence'],
                'top_classes': prediction_result['top_classes'],
                'top_probs': prediction_result['top_probs']
            },
            'model_info': {
                'total_classes': len(class_names),
                'model_type': 'MobileNetV2 Transfer Learning',
                'tensorflow_version': tf.__version__,
                'preprocessing': 'Colab-compatible'
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'debug_info': {
                'model_loaded': model is not None,
                'classes_loaded': class_names is not None
            }
        }), 500

@app.route('/predict_debug', methods=['POST'])
def predict_debug():
    """Debug endpoint to compare with Colab results"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Process image
        img_array = preprocess_image_colab_style(data['image'])
        
        # Get raw model output
        raw_predictions = model.predict(img_array, verbose=0)
        
        # Process predictions
        if len(raw_predictions.shape) > 1:
            prediction_probs = raw_predictions[0]
        else:
            prediction_probs = raw_predictions
        
        # Get detailed debug info
        top_5_indices = np.argsort(prediction_probs)[::-1][:5]
        
        debug_info = {
            'image_shape': img_array.shape,
            'image_dtype': str(img_array.dtype),
            'image_range': [float(img_array.min()), float(img_array.max())],
            'raw_predictions_shape': raw_predictions.shape,
            'predictions_sum': float(prediction_probs.sum()),
            'top_5_raw': [
                {
                    'index': int(idx),
                    'class': class_names[idx] if idx < len(class_names) else 'INVALID',
                    'raw_value': float(prediction_probs[idx])
                }
                for idx in top_5_indices
            ]
        }
        
        return jsonify({
            'success': True,
            'debug_info': debug_info,
            'model_info': {
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'total_classes': len(class_names)
            }
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'debug': True
        }), 500

# Giữ nguyên các routes khác...
@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all available food classes"""
    try:
        if class_names is None:
            return jsonify({'error': 'Classes not loaded'}), 503
        
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
        logger.info("🚀 API ready with Colab-compatible preprocessing!")
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
    
    logger.info(f"🌟 Starting Colab-compatible server on port {port}")
    logger.info(f"🔧 TensorFlow version: {tf.__version__}")
    app.run(host='0.0.0.0', port=port, debug=debug)