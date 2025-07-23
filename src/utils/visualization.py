import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
import tensorflow as tf

class ModelVisualizer:
    """Class for visualizing model performance and results."""
    
    @staticmethod
    def plot_training_history(history):
        """Plot training and validation accuracy/loss over epochs."""
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        
        epochs = range(1, len(acc) + 1)
        
        plt.figure(figsize=(12, 5))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names):
        """Plot a confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true, y_pred_prob, class_names):
        """Plot ROC curves for each class."""
        # Binarize the output
        y_test_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Plot ROC curves
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'red', 'green']
        
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(
                fpr[i], 
                tpr[i], 
                color=color,
                label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
    
    @staticmethod
    def plot_precision_recall_curve(y_true, y_pred_prob, class_names):
        """Plot precision-recall curves for each class."""
        # Binarize the output
        y_test_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        # Compute precision-recall curve and area for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(len(class_names)):
            precision[i], recall[i], _ = precision_recall_curve(
                y_test_bin[:, i], y_pred_prob[:, i]
            )
            average_precision[i] = average_precision_score(
                y_test_bin[:, i], y_pred_prob[:, i]
            )
        
        # Plot precision-recall curves
        plt.figure(figsize=(8, 6))
        colors = ['blue', 'red', 'green']
        
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(
                recall[i],
                precision[i],
                color=color,
                label=f'Precision-Recall curve of class {class_names[i]} (AP = {average_precision[i]:.2f})'
            )
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()
    
    @staticmethod
    def visualize_predictions(images, true_labels, pred_labels, class_names, num_images=9):
        """Visualize model predictions on sample images."""
        plt.figure(figsize=(12, 12))
        
        for i in range(min(num_images, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            
            true_label = class_names[true_labels[i]]
            pred_label = class_names[pred_labels[i]]
            
            color = 'green' if true_label == pred_label else 'red'
            
            plt.title(f'True: {true_label}\nPred: {pred_label}', color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()


def plot_class_distribution(y, class_names, title='Class Distribution'):
    """Plot the distribution of classes."""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y)
    plt.xticks(ticks=range(len(class_names)), labels=class_names, rotation=45)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of samples')
    plt.tight_layout()
    plt.show()
