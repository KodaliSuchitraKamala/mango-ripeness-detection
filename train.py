import os
import sys
import argparse
import numpy as np
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent))

from src.data.preprocess import DataPreprocessor
from src.models.model import MangoRipenessModel
from src.utils.visualization import ModelVisualizer, plot_class_distribution

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a mango ripeness classification model.')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/mango_images',
                      help='Path to the directory containing the dataset')
    parser.add_argument('--img-size', type=int, nargs=2, default=[224, 224],
                      help='Image size (height, width)')
    parser.add_argument('--test-size', type=float, default=0.15,
                      help='Proportion of data to use for testing')
    parser.add_argument('--val-size', type=float, default=0.15,
                      help='Proportion of training data to use for validation')
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for training')
    
    # Model arguments
    parser.add_argument('--model-name', type=str, default='efficientnetb0',
                      help='Base model architecture')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                      help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--fine-tune-epochs', type=int, default=20,
                      help='Number of fine-tuning epochs')
    
    # Output arguments
    parser.add_argument('--output-dir', type=str, default='output',
                      help='Directory to save model and results')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data preprocessor
    print("Initializing data preprocessor...")
    preprocessor = DataPreprocessor(
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        test_size=args.test_size,
        val_size=args.val_size
    )
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = preprocessor.load_data()
    
    # Plot class distribution before augmentation
    print("Original class distribution:")
    plot_class_distribution(y, preprocessor.classes, 'Original Class Distribution')
    
    # Augment data to balance classes
    print("Augmenting data to balance classes...")
    X_augmented, y_augmented = preprocessor.augment_data(X, y)
    
    # Plot class distribution after augmentation
    print("Class distribution after augmentation:")
    plot_class_distribution(y_augmented, preprocessor.classes, 'Class Distribution After Augmentation')
    
    # Split data into train, validation, and test sets
    print("Splitting data into train/val/test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X_augmented, y_augmented)
    
    # Calculate class weights
    class_weights = preprocessor.get_class_weights(y_train)
    print(f"Class weights: {class_weights}")
    
    # Create data generators
    print("Creating data generators...")
    train_generator, val_generator = preprocessor.create_data_generators(
        X_train, y_train, X_val, y_val, batch_size=args.batch_size
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(X_train) // args.batch_size
    validation_steps = len(X_val) // args.batch_size
    
    # Initialize model
    print("Initializing model...")
    model = MangoRipenessModel(
        input_shape=(args.img_size[0], args.img_size[1], 3),
        num_classes=len(preprocessor.classes),
        learning_rate=args.learning_rate
    )
    
    # Train the model
    print(f"Training model for {args.epochs} epochs...")
    history = model.train(
        train_generator,
        val_generator,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weights=class_weights
    )
    
    # Fine-tune the model
    print("Fine-tuning the model...")
    # Unfreeze the base model for fine-tuning
    model.model.layers[4].trainable = True
    
    # Recompile with a lower learning rate
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate / 10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training with fine-tuning
    history_fine = model.train(
        train_generator,
        val_generator,
        epochs=args.epochs + args.fine_tune_epochs,
        initial_epoch=history.epoch[-1] + 1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        class_weights=class_weights
    )
    
    # Evaluate the model on the test set
    print("Evaluating model on test set...")
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    test_generator = test_datagen.flow(
        X_test, y_test,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions on test set
    y_pred_prob = model.model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Generate and save visualizations
    print("Generating visualizations...")
    visualizer = ModelVisualizer()
    
    # Plot training history
    visualizer.plot_training_history(history_fine)
    
    # Plot confusion matrix
    visualizer.plot_confusion_matrix(y_test, y_pred, preprocessor.classes)
    
    # Plot ROC curves
    visualizer.plot_roc_curve(y_test, y_pred_prob, preprocessor.classes)
    
    # Plot precision-recall curves
    visualizer.plot_precision_recall_curve(y_test, y_pred_prob, preprocessor.classes)
    
    # Visualize sample predictions
    sample_indices = np.random.choice(len(X_test), size=min(9, len(X_test)), replace=False)
    visualizer.visualize_predictions(
        X_test[sample_indices],
        y_test[sample_indices],
        y_pred[sample_indices],
        preprocessor.classes
    )
    
    # Save the model
    model_path = os.path.join(args.output_dir, 'mango_ripeness_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Print classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=preprocessor.classes))
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    main()
