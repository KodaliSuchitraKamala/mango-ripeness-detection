import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageEnhance
import cv2
import io
import os
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model directly
def create_simple_model():
    """Create a simple CNN model for testing purposes."""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: overRipe, ripe, unRipe
    ])
    
    # Compile with default settings for testing
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

@st.cache_resource
def load_model():
    try:
        # Define the model path
        model_path = os.path.join('models', 'mango_ripeness_model.h5')
        
        # Check if model file exists
        if not os.path.exists(model_path):
            st.warning("Pre-trained model not found. Creating a simple model for testing...")
            model = create_simple_model()
            
            # Save the model for future use
            os.makedirs('models', exist_ok=True)
            model.save(model_path)
            st.success(f"Simple model created and saved to {os.path.abspath(model_path)}")
            return model
            
        # Load the existing model
        model = tf.keras.models.load_model(model_path)
        st.success("Pre-trained model loaded successfully!")
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Page configuration
st.set_page_config(
    page_title="Mango Ripeness Detector",
    page_icon="🥭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stRadio>div {
        flex-direction: row !important;
        gap: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def enhance_image_quality(image):
    """Enhance image quality for better prediction."""
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert to float32 for processing
    img_float = image.astype(np.float32) / 255.0
    
    # Convert to HSV color space for better color processing
    hsv = cv2.cvtColor(img_float, cv2.COLOR_RGB2HSV)
    
    # Adjust saturation and value (increase contrast)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.2, 0, 1)  # Increase saturation
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 1)  # Increase brightness
    
    # Convert back to RGB
    enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Apply sharpening
    kernel = np.array([[-1,-1,-1], 
                      [-1, 9,-1],
                      [-1,-1,-1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to 8-bit
    enhanced = (np.clip(enhanced, 0, 1) * 255).astype(np.uint8)
    
    return Image.fromarray(enhanced)

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess the image for model prediction with enhanced preprocessing."""
    try:
        # Ensure image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Enhance image quality
        enhanced_image = enhance_image_quality(image)
        
        # Resize and normalize
        resized_image = enhanced_image.resize(target_size)
        image_array = np.array(resized_image) / 255.0
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
        
    except Exception as e:
        st.warning(f"Image preprocessing warning: {str(e)}")
        # Fallback to basic preprocessing if enhancement fails
        image = image.convert('RGB').resize(target_size)
        return np.expand_dims(np.array(image) / 255.0, axis=0)

def get_ripeness_recommendation(ripeness_class, confidence):
    """Get detailed recommendation based on ripeness class and confidence."""
    recommendations = {
        'unRipe': {
            'title': '🍏 Not Ripe Yet',
            'message': 'This mango is not yet ripe.',
            'tips': [
                'Store at room temperature for 3-5 days',
                'Check daily for ripeness',
                'Ripe when slightly soft to touch and fragrant'
            ]
        },
        'ripe': {
            'title': '🥭 Perfectly Ripe!',
            'message': 'This mango is at peak ripeness!',
            'tips': [
                'Best eaten within 1-2 days',
                'Store in refrigerator to slow ripening',
                'Great for fresh eating or in salads'
            ]
        },
        'overRipe': {
            'title': '⚠️ Overripe',
            'message': 'This mango is past its prime.',
            'tips': [
                'Best used immediately in smoothies or baking',
                'Check for any signs of spoilage',
                'Can be frozen for later use in recipes'
            ]
        }
    }
    
    # Adjust message based on confidence
    if confidence < 60:
        recommendations[ripeness_class]['message'] += " (Low confidence prediction)"
    
    return recommendations[ripeness_class]

def predict_ripeness(model, image):
    """Make prediction using the loaded model with enhanced processing."""
    try:
        # Define class names in the correct order based on model training
        CLASS_NAMES = ['overRipe', 'ripe', 'unRipe']
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)[0]
        
        # Get the predicted class index and confidence
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx]) * 100
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get all predictions with their confidence scores
        all_predictions = [
            {'class': CLASS_NAMES[i], 'confidence': float(score) * 100}
            for i, score in enumerate(predictions)
        ]
        
        # Sort predictions by confidence (highest first)
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Get detailed recommendation
        recommendation = get_ripeness_recommendation(predicted_class, confidence)
        
        return {
            'class': predicted_class,
            'confidence': round(confidence, 2),
            'all_predictions': all_predictions,
            'recommendation': recommendation
        }
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def display_prediction_results(result):
    """Display prediction results in a user-friendly way."""
    if not result:
        return
    
    # Display the main prediction
    st.markdown("## 🎯 Prediction Results")
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Display prediction with appropriate icon and color
        if result['class'] == 'ripe':
            st.success(f"### 🥭 {result['class'].title()} (Confidence: {result['confidence']:.1f}%)")
        elif result['class'] == 'unRipe':
            st.info(f"### 🍏 {result['class'].title()} (Confidence: {result['confidence']:.1f}%)")
        else:  # overRipe
            st.warning(f"### ⚠️ {result['class'].title()} (Confidence: {result['confidence']:.1f}%)")
        
        # Display confidence meter
        st.progress(min(100, int(result['confidence'])))
        st.caption(f"Confidence: {result['confidence']:.1f}%")
        
        # Display all predictions
        st.markdown("### 📊 Prediction Probabilities:")
        for pred in result['all_predictions']:
            st.write(f"- {pred['class'].title()}: {pred['confidence']:.1f}%")
    
    with col2:
        # Display detailed recommendation
        rec = result.get('recommendation', {})
        st.markdown(f"### {rec.get('title', 'Recommendation')}")
        
        # Create an expandable section for details
        with st.expander("ℹ️ View Details", expanded=True):
            st.write(rec.get('message', ''))
            
            st.markdown("#### 💡 Tips:")
            for tip in rec.get('tips', []):
                st.markdown(f"- {tip}")
    
    # Add some space
    st.markdown("---")

def get_test_images():
    """Return a dictionary of test images with their expected classes.
    
    The categories are mapped as follows from the dataset:
    - overRipe: Overripe mangoes
    - ripe: Ripe mangoes
    - unRipe: Unripe mangoes
    """
    # Define test images with their expected categories
    test_images = {
        # Unripe mangoes (green, firm)
        'unripe_1': {'path': 'test_images/unripe_1.jpg', 'class': 'unRipe'},
        'unripe_2': {'path': 'test_images/unripe_2.jpg', 'class': 'unRipe'},
        
        # Ripe mangoes (yellow/orange, slightly soft)
        'ripe_1': {'path': 'test_images/ripe_1.jpg', 'class': 'ripe'},
        'ripe_2': {'path': 'test_images/ripe_2.jpg', 'class': 'ripe'},
        
        # Overripe mangoes (dark spots, very soft, possibly bruised)
        'overripe_1': {'path': 'test_images/overripe_1.jpg', 'class': 'overRipe'},
        'overripe_2': {'path': 'test_images/overripe_2.jpg', 'class': 'overRipe'},
    }
    
    # Create test_images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Create placeholder images with appropriate colors
    for img_info in test_images.values():
        if not os.path.exists(img_info['path']):
            # Create colored placeholders based on ripeness
            if 'unripe' in img_info['path']:
                # Green for unripe
                color = (34, 139, 34)  # Forest green
            elif 'ripe' in img_info['path']:
                # Yellow/Orange for ripe
                color = (255, 165, 0)   # Orange
            else:  # overripe
                # Deep orange/red for overripe
                color = (255, 69, 0)    # Orange-red
                
            # Create and save placeholder image
            img = Image.new('RGB', (224, 224), color=color)
            img.save(img_info['path'])
    
    return test_images

def interpret_prediction(prediction, confidence):
    """Provide human-readable interpretation of the prediction."""
    ripeness_descriptions = {
        'unRipe': {
            'color': 'green',
            'firmness': 'very firm',
            'taste': 'sour and starchy',
            'use': 'best for pickling or cooking',
            'storage': 'can be stored at room temperature for several days'
        },
        'ripe': {
            'color': 'yellow/orange',
            'firmness': 'slightly soft',
            'taste': 'sweet and juicy',
            'use': 'perfect for eating fresh',
            'storage': 'best consumed within 1-2 days, or refrigerate to slow ripening'
        },
        'overRipe': {
            'color': 'dark orange/red with possible spots',
            'firmness': 'very soft',
            'taste': 'very sweet, possibly fermented',
            'use': 'best for smoothies, baking, or sauces',
            'storage': 'use immediately or freeze for later use'
        }
    }
    
    desc = ripeness_descriptions.get(prediction, {})
    
    return {
        'ripeness_level': prediction,
        'confidence': f"{confidence:.1f}%",
        'appearance': f"The mango appears {desc.get('color', 'unknown')} in color.",
        'texture': f"It feels {desc.get('firmness', 'unknown')} to the touch.",
        'taste': f"It will likely taste {desc.get('taste', 'unknown')}.",
        'recommended_use': f"Best used for: {desc.get('use', 'various culinary uses')}.",
        'storage_tip': f"Storage tip: {desc.get('storage', 'store appropriately')}.",
        'is_ready': prediction == 'ripe',
        'needs_attention': prediction == 'overRipe'
    }

def analyze_prediction_errors(results):
    """Analyze prediction errors to identify patterns."""
    if not results:
        return "No results to analyze."
    
    errors = [r for r in results if not r['is_correct']]
    if not errors:
        return "No prediction errors to analyze. All tests passed!"
    
    analysis = ["## 🧐 Error Analysis\n"]
    
    # Group errors by expected class
    error_groups = {}
    for error in errors:
        expected = error['expected']
        if expected not in error_groups:
            error_groups[expected] = []
        error_groups[expected].append(error)
    
    # Analyze each error group
    for expected, errs in error_groups.items():
        analysis.append(f"### Errors for {expected} mangoes:")
        
        # Count what these were misclassified as
        misclass_counts = {}
        for err in errs:
            pred = err['predicted']
            misclass_counts[pred] = misclass_counts.get(pred, 0) + 1
        
        for pred, count in misclass_counts.items():
            analysis.append(f"- {count} misclassified as {pred}")
            
            # Get example images for this type of error
            examples = [e for e in errs if e['predicted'] == pred][:2]
            for ex in examples:
                analysis.append(f"  - Example: {ex['name']} (confidence: {ex['confidence']:.1f}%)")
        
        # Add spacing between groups
        analysis.append("")
    
    return "\n".join(analysis)

def download_sample_images():
    """Download sample test images if they don't exist."""
    import urllib.request
    import os
    
    # Create test_images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Sample image URLs (replace with actual image URLs or local paths)
    sample_images = {
        'unripe_1.jpg': 'https://example.com/path/to/unripe1.jpg',
        'unripe_2.jpg': 'https://example.com/path/to/unripe2.jpg',
        'ripe_1.jpg': 'https://example.com/path/to/ripe1.jpg',
        'ripe_2.jpg': 'https://example.com/path/to/ripe2.jpg',
        'overripe_1.jpg': 'https://example.com/path/to/overripe1.jpg',
        'overripe_2.jpg': 'https://example.com/path/to/overripe2.jpg',
    }
    
    for filename, url in sample_images.items():
        filepath = os.path.join('test_images', filename)
        if not os.path.exists(filepath):
            try:
                urllib.request.urlretrieve(url, filepath)
            except:
                # If download fails, create a placeholder
                color = 'green' if 'unripe' in filename else 'yellow' if 'ripe' in filename else 'orange'
                img = Image.new('RGB', (224, 224), color=color)
                img.save(filepath)

def calculate_metrics(results):
    """Calculate various metrics from test results."""
    if not results:
        return {}
    
    # Basic metrics
    total = len(results)
    correct = sum(1 for r in results if r['is_correct'])
    accuracy = correct / total * 100
    
    # Per-class metrics
    classes = set(r['expected'] for r in results)
    class_metrics = {}
    
    for cls in classes:
        class_results = [r for r in results if r['expected'] == cls]
        class_correct = sum(1 for r in class_results if r['is_correct'])
        class_metrics[cls] = {
            'precision': class_correct / len(class_results) * 100 if class_results else 0,
            'count': len(class_results),
            'correct': class_correct
        }
    
    # Confidence metrics
    confidences = [r['confidence'] for r in results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'class_metrics': class_metrics
    }

def display_confusion_matrix(results):
    """Display a confusion matrix of the test results."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create confusion matrix
        classes = sorted(set(r['expected'] for r in results))
        cm = pd.crosstab(
            pd.Series([r['expected'] for r in results], name='Actual'),
            pd.Series([r['predicted'] for r in results], name='Predicted'),
            dropna=False
        ).reindex(index=classes, columns=classes, fill_value=0)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   cbar=False, square=True, 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Could not display confusion matrix: {str(e)}")

def display_test_results(model, results):
    """Display test results with detailed analysis and visualizations."""
    if not results:
        st.warning("No test results to display.")
        return
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Display overall metrics
    st.markdown("## 📊 Test Results Summary")
    
    # Overall metrics in columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", f"{metrics['accuracy']:.1f}%")
    with col2:
        st.metric("Average Confidence", f"{metrics['avg_confidence']:.1f}%")
    with col3:
        correct = sum(1 for r in results if r['is_correct'])
        total = len(results)
        st.metric("Correct Predictions", f"{correct} / {total}")
    
    # Confusion matrix
    st.markdown("### Confusion Matrix")
    display_confusion_matrix(results)
    
    # Per-class metrics
    st.markdown("### Per-class Performance")
    for cls, cls_metrics in metrics['class_metrics'].items():
        with st.expander(f"{cls} (Accuracy: {cls_metrics['precision']:.1f}%)"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Correct Predictions:** {cls_metrics['correct']}/{cls_metrics['count']}")
                st.write(f"**Average Confidence:** {sum(r['confidence'] for r in results if r['expected'] == cls) / cls_metrics['count']:.1f}%")
            with col2:
                # Show example images for this class
                examples = [r for r in results if r['expected'] == cls][:2]
                for ex in examples:
                    st.image(ex['image'], width=150, caption=f"Predicted: {ex['predicted']} ({ex['confidence']:.1f}%)")
    
    # Error analysis
    st.markdown("### 🔍 Error Analysis")
    error_analysis = analyze_prediction_errors(results)
    st.markdown(error_analysis, unsafe_allow_html=True)
    
    # Detailed predictions
    st.markdown("### 📋 Detailed Predictions")
    for result in results:
        with st.expander(f"{result['name']}: {result['expected']} → {result['predicted']} "
                       f"({'✅' if result['is_correct'] else '❌'})", 
                       expanded=False):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(result['image'], use_column_width=True)
            with col2:
                # Basic info
                st.write(f"**Expected:** {result['expected']}")
                st.write(f"**Predicted:** {result['predicted']}")
                st.write(f"**Confidence:** {result['confidence']:.1f}%")
                
                # Interpretation
                interpretation = interpret_prediction(result['predicted'], result['confidence'])
                
                if not result['is_correct']:
                    st.error("❌ Incorrect prediction")
                    st.write(f"The model thought this {result['expected']} mango was {result['predicted']}.")
                else:
                    st.success("✅ Correct prediction")
                
                # Show interpretation
                st.markdown("#### Ripeness Interpretation:")
                st.write(interpretation['appearance'])
                st.write(interpretation['texture'])
                st.write(interpretation['taste'])
                st.write(interpretation['recommended_use'])
                st.write(interpretation['storage_tip'])
                
                # Show model's confidence in all classes
                st.markdown("#### Model's Confidence in All Classes:")
                for pred in result['all_predictions']:
                    bar = "█" * int(pred['confidence'] / 5)
                    st.write(f"{pred['class']}: {pred['confidence']:.1f}% {bar}")

def run_test_suite(model):
    """Run tests on all sample images and display graphical results."""
    st.markdown("## 🧪 Mango Ripeness Test Results")
    
    # Download sample images if needed
    with st.spinner("Preparing test images..."):
        download_sample_images()
    
    test_images = get_test_images()
    results = []
    
    # Run all tests automatically
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, (name, img_info) in enumerate(test_images.items()):
        try:
            status_text.text(f"Testing {name}...")
            progress_bar.progress((i + 1) / len(test_images))
            
            image = Image.open(img_info['path'])
            result = predict_ripeness(model, image)
            
            if result:
                results.append({
                    'image': image,
                    'name': name,
                    'expected': img_info['class'],
                    'predicted': result['class'],
                    'confidence': result['confidence'],
                    'is_correct': result['class'] == img_info['class'],
                    'all_predictions': result['all_predictions']
                })
        except Exception as e:
            st.error(f"Error testing {name}: {str(e)}")
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Display results
    if results:
        display_test_results(model, results)
    else:
        st.warning("No test results to display. The tests may have failed.")

def plot_confidence_distribution(results):
    """Plot the distribution of confidence scores."""
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='confidence', hue='is_correct', 
                    bins=20, kde=True, multiple='stack')
        plt.title('Distribution of Confidence Scores')
        plt.xlabel('Confidence (%)')
        plt.ylabel('Count')
        plt.legend(title='Correct?', labels=['No', 'Yes'])
        
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Could not plot confidence distribution: {str(e)}")

def plot_class_distribution(results):
    """Plot the distribution of actual vs predicted classes."""
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(results)
        
        # Count actual and predicted classes
        actual_counts = df['expected'].value_counts().sort_index()
        predicted_counts = df['predicted'].value_counts().sort_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual distribution
        actual_counts.plot(kind='bar', color='skyblue', ax=ax1)
        ax1.set_title('Actual Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        
        # Predicted distribution
        predicted_counts.plot(kind='bar', color='lightgreen', ax=ax2)
        ax2.set_title('Predicted Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not plot class distribution: {str(e)}")

def plot_confidence_by_class(results):
    """Plot confidence scores by class."""
    try:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(results)
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='expected', y='confidence', 
                   hue='is_correct')
        plt.title('Confidence Distribution by Class')
        plt.xlabel('Expected Class')
        plt.ylabel('Confidence (%)')
        plt.legend(title='Correct?')
        
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Could not plot confidence by class: {str(e)}")

def plot_feature_importance(model, input_shape=(224, 224, 3)):
    """Plot feature importance using Grad-CAM (Gradient-weighted Class Activation Mapping)."""
    try:
        import tensorflow as tf
        import numpy as np
        import matplotlib.pyplot as plt
        
        # This is a simplified version - in practice, you'd need to implement Grad-CAM
        # or use a library like tf-keras-vis
        st.warning("Grad-CAM visualization requires additional setup. "
                  "This is a placeholder for the visualization.")
        
        # Placeholder visualization
        plt.figure(figsize=(8, 8))
        plt.text(0.5, 0.5, 'Grad-CAM Visualization\n(Requires implementation)', 
                ha='center', va='center')
        plt.axis('off')
        st.pyplot(plt)
        
    except Exception as e:
        st.warning(f"Could not generate feature importance visualization: {str(e)}")

def plot_roc_curve(results):
    """Plot ROC curves for each class (one-vs-rest)."""
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Convert results to one-hot encoding
        classes = sorted(set(r['expected'] for r in results))
        y_true = [r['expected'] for r in results]
        y_scores = []
        
        for r in results:
            # Create a dictionary of class to confidence
            class_scores = {}
            for pred in r['all_predictions']:
                class_scores[pred['class']] = pred['confidence'] / 100.0  # Convert to 0-1 range
            
            # Ensure all classes are present
            scores = [class_scores.get(cls, 0.0) for cls in classes]
            y_scores.append(scores)
        
        y_true = label_binarize(y_true, classes=classes)
        y_scores = np.array(y_scores)
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i, cls in enumerate(classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(len(classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'ROC curve of class {classes[i]} (area = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        
        st.pyplot(plt)
    except Exception as e:
        st.warning(f"Could not generate ROC curves: {str(e)}")

def display_test_results(model, results):
    """Display test results with detailed analysis and visualizations."""
    if not results:
        st.warning("No test results to display.")
        return
    
    # Display overall metrics
    st.markdown("## 📊 Test Results Summary")
    
    # Overall metrics in columns
    metrics = calculate_metrics(results)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", f"{metrics['accuracy']:.1f}%")
    with col2:
        st.metric("Average Confidence", f"{metrics['avg_confidence']:.1f}%")
    with col3:
        correct = sum(1 for r in results if r['is_correct'])
        total = len(results)
        st.metric("Correct Predictions", f"{correct} / {total}")
    
    # Add tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Class Distribution", 
        "📊 Confidence Analysis", 
        "🎯 ROC Curves", 
        "🔍 Error Analysis",
        "📋 Details"
    ])
    
    with tab1:
        st.markdown("### Class Distribution")
        plot_class_distribution(results)
        
        st.markdown("### Confusion Matrix")
        display_confusion_matrix(results)
    
    with tab2:
        st.markdown("### Confidence Distribution")
        plot_confidence_distribution(results)
        
        st.markdown("### Confidence by Class")
        plot_confidence_by_class(results)
    
    with tab3:
        st.markdown("### ROC Curves")
        plot_roc_curve(results)
    
    with tab4:
        st.markdown("### Error Analysis")
        error_analysis = analyze_prediction_errors(results)
        st.markdown(error_analysis)
        
        # Show example misclassifications
        errors = [r for r in results if not r['is_correct']]
        if errors:
            st.markdown("### Example Misclassifications")
            for error in errors[:3]:  # Show up to 3 examples
                with st.expander(f"{error['name']}: Expected {error['expected']}, Predicted {error['predicted']}"):
                    st.image(error['image'], width=200)
                    st.write(f"Confidence: {error['confidence']:.1f}%")
    
    with tab5:
        st.markdown("### Detailed Results")
        
        # Show all test results in a table
        results_data = []
        for r in results:
            results_data.append({
                'Test Image': r['name'],
                'Expected': r['expected'],
                'Predicted': r['predicted'],
                'Confidence': f"{r['confidence']:.1f}%",
                'Correct': '✅' if r['is_correct'] else '❌'
            })
        
        st.dataframe(
            pd.DataFrame(results_data),
            column_config={
                'Test Image': 'Test Image',
                'Expected': 'Expected',
                'Predicted': 'Predicted',
                'Confidence': 'Confidence',
                'Correct': 'Correct'
            },
            use_container_width=True,
            hide_index=True
        )
        
        # Show raw results
        if st.checkbox("Show raw results"):
            st.json([{k: v for k, v in r.items() if k != 'image'} for r in results])

def main():
    st.title("🥭 Mango Ripeness Detector")
    
    # Add a tab layout
    tab1, tab2 = st.tabs(["🔍 Detect Ripeness", "🧪 Test Mode"])
    
    with tab1:
        # Existing main app code
        st.write("Upload an image or take a photo to check the ripeness of a mango.")
        
        # Load the model
        model = load_model()
        if model is None:
            st.error("Failed to load the model. Please check if the model file exists.")
            return
        
        # Sidebar for input selection
        st.sidebar.title("📷 Image Input")
        input_method = st.sidebar.radio(
            "Choose input method:",
            ("📁 Upload Image", "📷 Use Camera")
        )
        
        image = None
        
        # Handle image input
        if input_method == "📁 Upload Image":
            uploaded_file = st.sidebar.file_uploader(
                "Choose a mango image",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
        else:
            # Camera capture
            st.sidebar.info("Click the button below to take a picture")
            picture = st.camera_input("Take a picture of the mango")
            if picture:
                image = Image.open(picture)
                
                # Display the captured image in the sidebar
                st.sidebar.image(
                    image,
                    caption="Captured Image",
                    use_column_width=True
                )
                
                # Add a button to retake the picture
                if st.sidebar.button("🔄 Retake Picture"):
                    st.experimental_rerun()
        
        # Process and display the image
        if image is not None:
            # Display the original image
            st.image(
                image,
                caption="Your Mango",
                use_column_width=True
            )
            
            # Make prediction when button is clicked
            if st.button("🔍 Analyze Ripeness", use_container_width=True):
                with st.spinner("Analyzing mango ripeness..."):
                    # Make prediction
                    result = predict_ripeness(model, image)
                    
                    if result:
                        # Display results
                        display_prediction_results(result)
                    else:
                        st.error("Failed to analyze the image. Please try again with a clearer image.")

    with tab2:
        # Test mode
        model = load_model()
        if model is None:
            st.error("Cannot run tests: Model failed to load")
            return
            
        st.write("Run tests on sample images to evaluate model performance.")
        
        if st.button("🚀 Run Tests"):
            with st.spinner("Running tests..."):
                run_test_suite(model)

if __name__ == "__main__":
    main()