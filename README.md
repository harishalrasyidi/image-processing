# 🖼️ Image Processing Compilation

A modern web application for digital image processing built with FastAPI and OpenCV.

## ✨ Features

### 🎯 Single Image Processing
- **Basic Operations**
  - RGB Array Display
  - Grayscale Conversion
  - Image Inversion

- **Analysis**
  - Histogram Analysis
  - Statistical Information
  - Histogram Equalization

- **Advanced Processing**
  - Canny Edge Detection
  - Chain Code Generation
  - Integral Projection
  - Convolution Operations
  - Zero Padding
  - Image Filtering

### 🔄 Dual Image Processing
- Bitwise AND Operation
- Bitwise XOR Operation
- Histogram Specification

### 👤 Special Features
- Face Dataset Management
- Image Compression Tools

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
pip
virtualenv
```

### Installation

1. Clone the repository
```bash
git clone <repository-url>
cd fastapi-opencv-26agustus
```

2. Create and activate virtual environment
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/MacOS
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000` in your browser.

## 💡 Usage Examples

### Single Image Processing
1. Upload an image
2. Select desired operations
3. Adjust parameters if needed
4. Click "Process Single Image"

### Dual Image Processing
1. Upload two images
2. Choose operation type
3. Click "Process Dual Images"

## 🛠️ Tech Stack
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Data visualization
- [Jinja2](https://jinja.palletsprojects.com/) - Template engine

## 📁 Project Structure
```
fastapi-opencv-26agustus/
├── main.py              # Main application file
├── requirements.txt     # Dependencies
├── static/             # Static files
│   ├── uploads/        # Uploaded images
│   ├── histograms/     # Generated histograms
│   ├── dataset/        # Face datasets
│   └── processed_dataset/ # Processed images
└── templates/          # HTML templates
    ├── base.html       # Base template
    ├── home.html       # Main page
    └── ...            # Other templates
```

## 🌟 Features in Detail

### Image Analysis
- **Histogram Analysis**: Generate and visualize image histograms
- **Statistical Information**: Calculate mean and standard deviation
- **Edge Detection**: Implement Canny edge detection algorithm

### Image Manipulation
- **Convolution Operations**: Apply various kernels (Average, Sharpen, Edge)
- **Filtering**: Low-pass, High-pass, and Band-pass filters
- **Zero Padding**: Add padding around images

### Face Processing
- Upload and manage face datasets
- Apply noise reduction
- Enhance face images

### Image Compression
- JPEG compression with quality control
- PNG compression with level selection
- Quality metrics (PSNR, SSIM)

## 📝 License
MIT License - feel free to use and modify for your projects!
