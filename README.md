# Digital Image Processing Web Application

A comprehensive web application for digital image processing and analysis, built with FastAPI and OpenCV. This application provides a wide range of image processing capabilities through an intuitive web interface.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Architecture](#architecture)
- [Usage Guide](#usage-guide)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features

### Module 1: Introduction to Digital Image Processing
- **Basic Image Operations**
  - Image Loading and Display
  - RGB Channel Separation
  - Color Image to Grayscale Conversion
  - Basic File Operations

### Module 2: Image Enhancement
- **Histogram Processing**
  - Histogram Generation and Display
  - Histogram Equalization
  - Histogram Specification
  - Statistical Analysis
    - Mean Calculation
    - Standard Deviation
    - Variance Analysis

### Module 3: Spatial Domain Processing
- **Convolution Operations**
  - Custom Kernel Application
  - Zero Padding Implementation
  - Various Filter Types:
    - Average Filter
    - Gaussian Filter
    - Median Filter
    - Sharpening Filter

### Module 4: Image Segmentation and Analysis
- **Edge Detection**
  - Canny Edge Detection
  - Parameter Control:
    - Kernel Size
    - Sigma Value
    - Threshold Levels
- **Chain Code**
  - Contour Detection
  - Freeman Chain Code Generation
  - Boundary Analysis
- **Integral Projection**
  - Horizontal Projection
  - Vertical Projection
  - Threshold Control

### Module 5: Advanced Image Processing
- **Image Compression**
  - JPEG Compression
    - Quality Control
    - Size Optimization
  - PNG Compression
    - Level Selection
    - Lossless Compression
  - Compression Analysis
    - PSNR Calculation
    - SSIM Measurement
    - Compression Ratio

### Module 6: Frequency Domain Processing
- **Fourier Transform**
  - DFT Implementation
  - Frequency Spectrum Display
  - Filtering in Frequency Domain
- **Image Restoration**
  - Noise Reduction
  - Image Deblurring
  - Frequency-based Enhancement

### Module 7: Color Space Processing
- **Color Space Conversions**
  - RGB Color Space
    - Channel Separation
    - Component Analysis
  - CIE Color Spaces
    - XYZ Color Space
    - LAB Color Space
  - Television Standards
    - YCbCr Color Space
    - YIQ Color Space
    - YUV Color Space
- **Component Analysis**
  - Channel Visualization
  - Luminance Processing
  - Chrominance Analysis

### Additional Features
- **Dual Image Processing**
  - Bitwise Operations
    - AND Operation
    - OR Operation
    - XOR Operation
  - Image Blending
  - Image Comparison

- **Batch Processing**
  - Multiple Image Processing
  - Bulk Operations
  - Batch Export

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (venv or conda)

### Setup Instructions

1. Clone the repository
```bash
git clone <repository-url>
cd fastapi-opencv-26agustus
```

2. Create and activate virtual environment
```bash
# Using venv
python -m venv venv

# Windows activation
.\venv\Scripts\activate

# Unix/MacOS activation
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Start the application
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

## Architecture

### Project Structure
```
fastapi-opencv-26agustus/
├── main.py                 # Application entry point
├── requirements.txt        # Project dependencies
├── static/                 # Static assets
│   ├── uploads/           # User uploaded images
│   ├── histograms/        # Generated histograms
│   ├── dataset/           # Face datasets
│   └── processed_dataset/ # Processed images
├── templates/             # HTML templates
│   ├── base.html         # Base template
│   ├── home.html         # Landing page
│   ├── color_space.html  # Color space conversion
│   └── ...               # Feature-specific templates
└── README.md             # Documentation
```

### Technology Stack
- **Backend Framework**: FastAPI
- **Image Processing**: OpenCV (cv2)
- **Numerical Computing**: NumPy
- **Data Visualization**: Matplotlib
- **Template Engine**: Jinja2
- **Frontend**: Bootstrap 5

## Usage Guide

### Single Image Processing
1. Navigate to the desired processing module
2. Upload an image using the file selector
3. Choose processing parameters
4. Click "Process" to apply changes
5. View and download results

### Color Space Conversion
1. Access the Color Space module
2. Upload an RGB image
3. Select target color space
4. View original image, converted result, and individual components

### Batch Processing
1. Upload multiple images
2. Select processing operations
3. Configure parameters
4. Process all images
5. Download results in bulk

## Development

### Adding New Features
1. Create new route in `main.py`
2. Add corresponding template in `templates/`
3. Implement processing logic
4. Update sidebar in `base.html`
5. Document in README.md

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Document functions and classes
- Write unit tests for new features

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- OpenCV Documentation
- FastAPI Community
- Scientific Python Community
