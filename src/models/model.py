import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard
)
import os
import datetime

class MangoRipenessModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=3, learning_rate=1e-4):
        """
        Initialize the Mango Ripeness Classification Model.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            num_classes (int): Number of output classes
            learning_rate (float): Learning rate for the optimizer
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the model architecture using transfer learning with EfficientNetB0."""
        # Load pre-trained EfficientNetB0 model without top layers
        base_model = applications.EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling='avg'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Create the model
        inputs = layers.Input(shape=self.input_shape)
        
        # Data augmentation layers
        x = layers.Rescaling(1./255)(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomBrightness(0.2)(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Additional layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.BatchNormalization()(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def get_callbacks(self, checkpoint_dir='checkpoints'):
        """Get training callbacks."""
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            # Save the best model during training
            ModelCheckpoint(
                filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            # Stop training if validation loss doesn't improve for 10 epochs
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            # Log metrics for TensorBoard
            TensorBoard(
                log_dir=os.path.join('logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def train(
        self, 
        train_generator, 
        val_generator, 
        epochs=50, 
        steps_per_epoch=None, 
        validation_steps=None,
        class_weights=None
    ):
        """
        Train the model.
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator
            epochs (int): Number of training epochs
            steps_per_epoch (int): Number of steps per epoch
            validation_steps (int): Number of validation steps
            class_weights (dict): Class weights for imbalanced data
            
        Returns:
            History object containing training/validation metrics
        """
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train the model
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        return history
    
    def evaluate(self, test_generator):
        """Evaluate the model on test data."""
        return self.model.evaluate(test_generator, verbose=1)
    
    def predict(self, image):
        """Make predictions on a single image."""
        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = tf.expand_dims(image, axis=0)
        
        # Make prediction
        predictions = self.model.predict(image, verbose=0)
        return predictions
    
    def save(self, filepath):
        """Save the model to a file."""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, **kwargs):
        """Load a saved model."""
        # Create model instance
        model = cls(**kwargs)
        
        # Load weights
        model.model = models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        
        return model
