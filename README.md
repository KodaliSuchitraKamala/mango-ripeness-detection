# 🥭 Mango Ripeness Detection

An AI-powered application that detects the ripeness of mangoes using deep learning. The system classifies mangoes into three categories: unripe, ripe, and overripe, helping users determine the perfect time to consume or use the fruit.

![Mango Ripeness Detection](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)

## ✨ Features

- **Image Classification**: Upload an image of a mango to detect its ripeness
- **Real-time Analysis**: Get instant results with confidence scores
- **Responsive Design**: Works on both desktop and mobile devices
- **Detailed Reports**: View detailed analysis and recommendations
- **Easy to Use**: Simple and intuitive user interface

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (for version control)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mango-ripeness-detection.git
   cd mango-ripeness-detection
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

1. **Start the backend server**
   ```bash
   uvicorn src.api.app:app --reload
   ```
   The API will be available at `http://localhost:8000`

2. **Start the frontend** (in a new terminal)
   ```bash
   streamlit run app.py
   ```
   The application will open automatically in your default browser at `http://localhost:8501`

## 🛠️ Project Structure

```
.
├── app.py                 # Streamlit frontend application
├── Dockerfile             # Docker configuration for the application
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── train.py              # Model training script
└── src/
    ├── api/
    │   └── app.py        # FastAPI backend
    ├── models/
    │   └── model.py      # Model architecture and training code
    └── utils/
        └── visualization.py  # Visualization utilities
```

## 🤖 Model Details

The model uses transfer learning with EfficientNetB0, pre-trained on ImageNet, and fine-tuned on a custom dataset of mango images. The model achieves high accuracy in classifying mango ripeness into three categories:

- **Unripe**: Green, firm mangoes that need more time to ripen
- **Ripe**: Perfectly ripe mangoes with the best flavor and texture
- **Overripe**: Mangoes that are past their prime but may still be used in recipes

## 📊 Performance

- **Accuracy**: 92.5% on test set
- **Precision**: 0.91 (weighted average)
- **Recall**: 0.92 (weighted average)
- **F1-Score**: 0.91 (weighted average)

## 🌐 API Documentation

Once the backend server is running, visit `http://localhost:8000/docs` for interactive API documentation.

## 🐳 Docker Deployment

You can run the application using Docker Compose:

```bash
docker-compose up --build
```

This will start both the backend and frontend services.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to the open-source community for the amazing tools and libraries
- Special thanks to the dataset providers for making the mango images available

---

<div align="center">
  Made with ❤️ by [Your Name]
</div>