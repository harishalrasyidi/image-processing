import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List
import io
import base64
from skimage.metrics import structural_similarity as ssim
import matplotlib
matplotlib.use('Agg')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

if not os.path.exists("static/dataset"):
    os.makedirs("static/dataset")
if not os.path.exists("static/processed_dataset"):
    os.makedirs("static/processed_dataset")

# Modul 1 - RGB Array Display
@app.get("/rgb_display/", response_class=HTMLResponse)
async def rgb_display_form(request: Request):
    return templates.TemplateResponse("rgb_display.html", {"request": request})

@app.post("/rgb_display/", response_class=HTMLResponse)
async def process_rgb_display(request: Request, image: UploadFile = File(...)):
    # Read and process the image
    image_data = await image.read()
    file_extension = image.filename.split(".")[-1]
    filename = f"{uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    # Save the uploaded image
    with open(file_path, "wb") as f:
        f.write(image_data)

    # Process the image to get RGB arrays
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    # Create RGB array dictionary
    rgb_array = {
        "R": r.tolist(),
        "G": g.tolist(),
        "B": b.tolist()
    }

    return templates.TemplateResponse("rgb_display.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}",
        "rgb_array": rgb_array
    })

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    file_path = save_image(img, "uploaded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": file_path,
        "modified_image_path": file_path
    })

@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "subtract":
        result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max":
        result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    operation: str = Form(...)
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    original_path = save_image(img1, "original")

    if operation == "not":
        result_img = cv2.bitwise_not(img1)
    else:
        if file2 is None:
            return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar.", status_code=400)
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

        if operation == "and":
            result_img = cv2.bitwise_and(img1, img2)
        elif operation == "xor":
            result_img = cv2.bitwise_xor(img1, img2)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })
@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    # Menampilkan form untuk upload gambar ke grayscale
    return templates.TemplateResponse("grayscale.html", {"request": request})

@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})

@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })



@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})

@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    equalized_img = cv2.equalizeHist(img)

    original_path = save_image(img, "original")
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })

@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})

@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)

    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    # Di sini Anda bisa menambahkan algoritma spesifikasi histogram sebenarnya
    specified_img = cv2.equalizeHist(ref_img)

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path
    })


@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })

@app.get("/convolution/", response_class=HTMLResponse)
async def convolution_form(request: Request):
    return templates.TemplateResponse("convolution.html", {"request": request})

@app.post("/convolution/", response_class=HTMLResponse)
async def apply_convolution(
    request: Request, 
    file: UploadFile = File(...),
    kernel_type: str = Form(...)
):
    # Baca gambar
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    # Definisi kernel
    if kernel_type == "average":
        kernel = np.ones((3,3), np.float32) / 9
    elif kernel_type == "sharpen":
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
    elif kernel_type == "edge":
        kernel = np.array([[-1,-1,-1],
                         [-1, 8,-1],
                         [-1,-1,-1]])
    
    # Terapkan konvolusi
    result = cv2.filter2D(img, -1, kernel)
    
    # Simpan gambar
    original_path = save_image(img, "original")
    convolved_path = save_image(result, "convolved")
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": convolved_path
    })

@app.get("/padding/", response_class=HTMLResponse)
async def padding_form(request: Request):
    return templates.TemplateResponse("padding.html", {"request": request})

@app.post("/padding/", response_class=HTMLResponse)
async def apply_padding(
    request: Request, 
    file: UploadFile = File(...),
    padding_size: int = Form(...)
):
    # Baca gambar
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    # Terapkan zero padding
    result = cv2.copyMakeBorder(
        img,
        padding_size, padding_size, padding_size, padding_size,
        cv2.BORDER_CONSTANT,
        value=[0,0,0]
    )
    
    # Simpan gambar
    original_path = save_image(img, "original")
    padded_path = save_image(result, "padded")
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": padded_path
    })

@app.get("/filter/", response_class=HTMLResponse)
async def filter_form(request: Request):
    return templates.TemplateResponse("filter.html", {"request": request})

@app.post("/filter/", response_class=HTMLResponse)
async def apply_filter(
    request: Request, 
    file: UploadFile = File(...),
    filter_type: str = Form(...),
    kernel_size: int = Form(...)
):
    # Baca gambar
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if filter_type == "lowpass":
        # Gaussian blur untuk low pass filter
        result = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    elif filter_type == "highpass":
        # High pass filter dengan laplacian
        blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        result = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
    elif filter_type == "bandpass":
        # Band pass filter (kombinasi low dan high pass)
        blur1 = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        blur2 = cv2.GaussianBlur(img, (kernel_size*2, kernel_size*2), 0)
        result = cv2.subtract(blur1, blur2)
    
    # Simpan gambar
    original_path = save_image(img, "original")
    filtered_path = save_image(result, "filtered")
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": filtered_path
    })

def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"

def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"

def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"

# modul 4
def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    return faces

def add_salt_and_pepper(image, prob=0.02):
    output = np.copy(image)
    salt = np.random.random(image.shape) < prob/2
    output[salt] = 255
    pepper = np.random.random(image.shape) < prob/2
    output[pepper] = 0
    return output

def remove_noise(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def sharpen_image(image):
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    return cv2.filter2D(image, -1, kernel)

def process_image(image):
    noisy_image = add_salt_and_pepper(image)
    denoised_image = remove_noise(noisy_image)
    sharpened_image = sharpen_image(denoised_image)
    return noisy_image, denoised_image, sharpened_image

@app.get("/face_dataset/", response_class=HTMLResponse)
async def face_dataset_form(request: Request):
    return templates.TemplateResponse("face_dataset.html", {"request": request})

@app.post("/add_face/")
async def add_face(request: Request, new_person: str = Form(...), images: List[UploadFile] = File(...)):
    save_path = os.path.join('static/dataset', new_person)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    saved_images = []
    for i, file in enumerate(images):
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is not None:
            faces = detect_faces(img)
            for j, (x, y, w, h) in enumerate(faces):
                face = img[y:y+h, x:x+w]
                img_path = os.path.join(save_path, f"img_{i}_{j}.jpg")
                cv2.imwrite(img_path, face)
                saved_images.append(img_path)
    
    return templates.TemplateResponse("face_dataset.html", {
        "request": request,
        "message": f"Successfully saved {len(saved_images)} face images for {new_person}"
    })

@app.get("/process_images/", response_class=HTMLResponse)
async def process_images_form(request: Request):
    dataset_path = "static/dataset"
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    return templates.TemplateResponse("process_images.html", {
        "request": request,
        "persons": persons,
        "show_results": False
    })

@app.post("/process_images/", response_class=HTMLResponse)
async def process_images(request: Request, selected_person: str = Form(...)):
    person_path = os.path.join('static/dataset', selected_person)
    processed_path = os.path.join('static/processed_dataset', selected_person)
    
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
    
    images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if images:
        # Process first image for display
        img_path = os.path.join(person_path, images[0])
        img = cv2.imread(img_path)
        
        # Process and save all images
        noisy_image, denoised_image, sharpened_image = process_image(img)
        
        # Save processed images
        base_name = os.path.splitext(images[0])[0]
        cv2.imwrite(os.path.join(processed_path, f"{base_name}_noisy.jpg"), noisy_image)
        cv2.imwrite(os.path.join(processed_path, f"{base_name}_denoised.jpg"), denoised_image)
        cv2.imwrite(os.path.join(processed_path, f"{base_name}_sharpened.jpg"), sharpened_image)
        
        # Process remaining images in background
        for img_name in images[1:]:
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                _, _, sharpened = process_image(img)
                base_name = os.path.splitext(img_name)[0]
                cv2.imwrite(os.path.join(processed_path, f"{base_name}_processed.jpg"), sharpened)
        
        return templates.TemplateResponse("process_images.html", {
            "request": request,
            "persons": [d for d in os.listdir('static/dataset') if os.path.isdir(os.path.join('static/dataset', d))],
            "show_results": True,
            "original_image": f"/static/dataset/{selected_person}/{images[0]}",
            "noisy_image": f"/static/processed_dataset/{selected_person}/{base_name}_noisy.jpg",
            "denoised_image": f"/static/processed_dataset/{selected_person}/{base_name}_denoised.jpg",
            "sharpened_image": f"/static/processed_dataset/{selected_person}/{base_name}_sharpened.jpg"
        })
    
    return templates.TemplateResponse("process_images.html", {
        "request": request,
        "persons": [d for d in os.listdir('static/dataset') if os.path.isdir(os.path.join('static/dataset', d))],
        "show_results": False,
        "error": "No images found for selected person"
    })

# modul 5
def generate_freeman_chain_code(contour):
    """Generate Freeman 8-direction Chain Code from OpenCV contour."""
    chain_code = []
    if len(contour) < 2:
        return chain_code
    
    directions = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    
    for i in range(len(contour) - 1):
        p1 = contour[i][0]
        p2 = contour[i + 1][0]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        norm_dx = np.sign(dx)
        norm_dy = np.sign(dy)
        direction_key = (norm_dx, norm_dy)
        if direction_key in directions:
            chain_code.append(directions[direction_key])
    
    return chain_code

def calculate_projections(binary_norm):
    """Calculate horizontal and vertical projections."""
    horizontal_projection = np.sum(binary_norm, axis=0)
    vertical_projection = np.sum(binary_norm, axis=1)
    return horizontal_projection, vertical_projection

def save_plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def calculate_compression_metrics(original, compressed):
    # Calculate PSNR
    mse = np.mean((original.astype(float) - compressed.astype(float)) ** 2)
    psnr = float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
    
    # Calculate SSIM
    is_color = len(original.shape) == 3
    min_dim = min(original.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3
        
    ssim_value = ssim(
        original, compressed,
        channel_axis=2 if is_color else None,
        win_size=win_size,
        data_range=original.max() - original.min()
    )
    
    return psnr, ssim_value

def create_quality_comparison_plot(original_size, compressed_sizes, psnrs, ssims, method):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    qualities = [95, 75, 50, 25, 10] if method == "JPEG" else list(range(10))
    
    # Plot file sizes
    ax1.plot(qualities, compressed_sizes, 'b-o')
    ax1.axhline(y=original_size, color='r', linestyle='--', label='Original Size')
    ax1.set_xlabel('Quality Level' if method == "JPEG" else 'Compression Level')
    ax1.set_ylabel('File Size (KB)')
    ax1.set_title('File Size Comparison')
    ax1.legend()
    ax1.grid(True)
    
    # Plot quality metrics
    ax2.plot(qualities, psnrs, 'b-o', label='PSNR')
    ax2.set_xlabel('Quality Level' if method == "JPEG" else 'Compression Level')
    ax2.set_ylabel('PSNR (dB)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    ax3 = ax2.twinx()
    ax3.plot(qualities, ssims, 'r-o', label='SSIM')
    ax3.set_ylabel('SSIM', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title('Quality Metrics')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    return save_plot_to_image(fig)

@app.get("/canny_edge/", response_class=HTMLResponse)
async def canny_edge_form(request: Request):
    return templates.TemplateResponse("canny_edge.html", {"request": request})

@app.post("/canny_edge/")
async def process_canny_edge(
    request: Request,
    image: UploadFile = File(...),
    kernel_size: int = Form(5),
    sigma: float = Form(1.0),
    low_threshold: int = Form(50),
    high_threshold: int = Form(150)
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), sigma)
    
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Save images
    cv2.imwrite("static/uploads/original.jpg", gray)
    cv2.imwrite("static/uploads/blurred.jpg", blurred)
    cv2.imwrite("static/uploads/edges.jpg", edges)
    
    # Analyze edge distribution
    edge_pixels = np.sum(edges == 255)
    total_pixels = edges.size
    edge_percentage = (edge_pixels / total_pixels) * 100
    
    # Create distribution chart
    height, width = edges.shape
    regions = [
        edges[:height//2, :width//2],
        edges[:height//2, width//2:],
        edges[height//2:, :width//2],
        edges[height//2:, width//2:]
    ]
    region_names = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"]
    region_edges = [np.sum(region == 255) for region in regions]
    
    plt.figure()
    plt.pie(region_edges, labels=region_names, autopct='%1.1f%%')
    plt.savefig('static/uploads/distribution.png')
    plt.close()
    
    return templates.TemplateResponse("canny_edge.html", {
        "request": request,
        "show_results": True,
        "original_image": "/static/uploads/original.jpg",
        "blurred_image": "/static/uploads/blurred.jpg",
        "edge_image": "/static/uploads/edges.jpg",
        "distribution_chart": "/static/uploads/distribution.png",
        "edge_analysis": {
            "edge_pixels": edge_pixels,
            "total_pixels": total_pixels,
            "edge_percentage": edge_percentage
        }
    })

@app.get("/chain_code/", response_class=HTMLResponse)
async def chain_code_form(request: Request):
    return templates.TemplateResponse("chain_code.html", {"request": request})

@app.post("/chain_code/")
async def process_chain_code(
    request: Request,
    image: UploadFile = File(...),
    threshold_type: str = Form(...),
    threshold_value: int = Form(127),
    contour_mode: str = Form("RETR_EXTERNAL")
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binarization
    thresh_mode = cv2.THRESH_BINARY if threshold_type == "BINARY" else cv2.THRESH_BINARY_INV
    _, binary = cv2.threshold(gray, threshold_value, 255, thresh_mode)
    
    # Contour detection
    contour_mode_map = {
        "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
        "RETR_LIST": cv2.RETR_LIST,
        "RETR_CCOMP": cv2.RETR_CCOMP,
        "RETR_TREE": cv2.RETR_TREE
    }
    contours, _ = cv2.findContours(binary, contour_mode_map[contour_mode], cv2.CHAIN_APPROX_NONE)
    
    if len(contours) > 0:
        # Find largest contour
        largest_contour_idx = np.argmax([cv2.contourArea(c) for c in contours])
        largest_contour = contours[largest_contour_idx]
        
        # Generate chain code
        chain_code = generate_freeman_chain_code(largest_contour)
        
        # Create visualization
        img_contour = img.copy()
        cv2.drawContours(img_contour, contours, -1, (100, 200, 255), 1)
        cv2.drawContours(img_contour, [largest_contour], 0, (0, 255, 0), 2)
        
        # Save images
        cv2.imwrite("static/uploads/original.jpg", gray)
        cv2.imwrite("static/uploads/binary.jpg", binary)
        cv2.imwrite("static/uploads/contours.jpg", img_contour)
        
        # Create histogram
        plt.figure(figsize=(10, 4))
        bins = np.arange(0, 9) - 0.5
        plt.hist(chain_code, bins=bins, rwidth=0.8)
        plt.xticks(range(8))
        plt.xlabel("Direction Code")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig('static/uploads/histogram.png')
        plt.close()
        
        return templates.TemplateResponse("chain_code.html", {
            "request": request,
            "show_results": True,
            "original_image": "/static/uploads/original.jpg",
            "binary_image": "/static/uploads/binary.jpg",
            "contour_image": "/static/uploads/contours.jpg",
            "histogram_image": "/static/uploads/histogram.png",
            "contour_length": len(largest_contour),
            "chain_code_length": len(chain_code),
            "chain_code": ", ".join(map(str, chain_code))
        })
    
    return templates.TemplateResponse("chain_code.html", {
        "request": request,
        "show_results": False,
        "error": "No contours found"
    })

@app.get("/integral_projection/", response_class=HTMLResponse)
async def integral_projection_form(request: Request):
    return templates.TemplateResponse("integral_projection.html", {"request": request})

@app.post("/integral_projection/")
async def process_integral_projection(
    request: Request,
    image: UploadFile = File(...),
    threshold_type: str = Form(...),
    threshold_value: int = Form(127),
    apply_blur: bool = Form(False),
    blur_kernel_size: int = Form(5)
):
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if apply_blur:
        gray = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    
    if threshold_type == "OTSU":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    else:
        thresh_mode = cv2.THRESH_BINARY if threshold_type == "BINARY" else cv2.THRESH_BINARY_INV
        _, binary = cv2.threshold(gray, threshold_value, 255, thresh_mode)
    
    binary_norm = binary / 255.0
    h_proj, v_proj = calculate_projections(binary_norm)
    
    # Create main visualization
    height, width = binary_norm.shape
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                         left=0.1, right=0.9, bottom=0.1, top=0.9,
                         wspace=0.05, hspace=0.05)
    
    # Original image
    ax_orig = plt.subplot(gs[0, 0])
    ax_orig.imshow(gray, cmap='gray')
    ax_orig.set_title('Original Image')
    ax_orig.axis('off')
    
    # Binary image
    ax_img = plt.subplot(gs[1, 0])
    ax_img.imshow(binary_norm, cmap='gray')
    ax_img.set_title('Binary Image')
    ax_img.set_xlabel('Column Index')
    ax_img.set_ylabel('Row Index')
    
    # Horizontal Projection
    ax_hproj = plt.subplot(gs[0, 1], sharey=ax_orig)
    ax_hproj.plot(h_proj, np.arange(width), color='blue')
    ax_hproj.set_title('Horizontal Projection')
    ax_hproj.invert_xaxis()
    plt.setp(ax_hproj.get_yticklabels(), visible=False)
    
    # Vertical Projection
    ax_vproj = plt.subplot(gs[1, 1], sharex=ax_hproj)
    ax_vproj.plot(v_proj, np.arange(height), color='red')
    ax_vproj.set_title('Vertical Projection')
    ax_vproj.invert_yaxis()
    plt.setp(ax_vproj.get_xticklabels(), visible=False)
    
    plt.savefig('static/uploads/main_visualization.png')
    plt.close()
    
    # Create individual projection plots
    plt.figure(figsize=(10, 4))
    plt.plot(h_proj, color='blue')
    plt.title("Horizontal Projection")
    plt.xlabel("Column Index")
    plt.ylabel("Sum of Pixels")
    plt.grid(True, alpha=0.3)
    plt.savefig('static/uploads/h_proj.png')
    plt.close()
    
    plt.figure(figsize=(10, 4))
    plt.plot(v_proj, color='red')
    plt.title("Vertical Projection")
    plt.xlabel("Row Index")
    plt.ylabel("Sum of Pixels")
    plt.grid(True, alpha=0.3)
    plt.savefig('static/uploads/v_proj.png')
    plt.close()
    
    return templates.TemplateResponse("integral_projection.html", {
        "request": request,
        "show_results": True,
        "main_visualization": "/static/uploads/main_visualization.png",
        "h_proj_plot": "/static/uploads/h_proj.png",
        "v_proj_plot": "/static/uploads/v_proj.png",
        "h_proj_size": len(h_proj),
        "h_proj_max": f"{np.max(h_proj):.2f}",
        "h_proj_mean": f"{np.mean(h_proj):.2f}",
        "v_proj_size": len(v_proj),
        "v_proj_max": f"{np.max(v_proj):.2f}",
        "v_proj_mean": f"{np.mean(v_proj):.2f}"
    })

# Modul 6
@app.get("/compression/", response_class=HTMLResponse)
async def compression_form(request: Request):
    return templates.TemplateResponse("compression.html", {"request": request})

@app.post("/compression/")
async def process_compression(
    request: Request,
    image: UploadFile = File(...),
    method: str = Form(...),
    jpeg_quality: int = Form(None),
    png_level: int = Form(None)
):
    # Read the image
    image_data = await image.read()
    np_array = np.frombuffer(image_data, np.uint8)
    original_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if original_img is None:
        return HTMLResponse("Failed to read the uploaded image", status_code=400)
    
    # Create temporary files for compression
    temp_original = f"static/uploads/temp_original_{uuid4()}.png"
    temp_compressed = f"static/uploads/temp_compressed_{uuid4()}.{method.lower()}"
    
    # Save original image
    cv2.imwrite(temp_original, original_img)
    original_size = os.path.getsize(temp_original) / 1024  # KB
    
    # Compress the image
    if method == "JPEG":
        quality = jpeg_quality if jpeg_quality is not None else 95
        cv2.imwrite(temp_compressed, original_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:  # PNG
        level = png_level if png_level is not None else 6
        cv2.imwrite(temp_compressed, original_img, [cv2.IMWRITE_PNG_COMPRESSION, level])
    
    # Read back the compressed image
    compressed_img = cv2.imread(temp_compressed)
    compressed_size = os.path.getsize(temp_compressed) / 1024  # KB
    
    # Calculate metrics
    psnr, ssim_value = calculate_compression_metrics(original_img, compressed_img)
    
    # Generate comparison plots if JPEG
    comparison_plot = None
    if method == "JPEG":
        qualities = [95, 75, 50, 25, 10]
        compressed_sizes = []
        psnrs = []
        ssims = []
        
        for q in qualities:
            temp_file = f"static/uploads/temp_{q}.jpg"
            cv2.imwrite(temp_file, original_img, [cv2.IMWRITE_JPEG_QUALITY, q])
            temp_img = cv2.imread(temp_file)
            temp_psnr, temp_ssim = calculate_compression_metrics(original_img, temp_img)
            
            compressed_sizes.append(os.path.getsize(temp_file) / 1024)
            psnrs.append(temp_psnr)
            ssims.append(temp_ssim)
            
            os.remove(temp_file)
            
        comparison_plot = create_quality_comparison_plot(
            original_size, compressed_sizes, psnrs, ssims, method
        )
    
    # Convert images to base64 for display
    _, original_buffer = cv2.imencode('.png', original_img)
    _, compressed_buffer = cv2.imencode('.png', compressed_img)
    
    original_base64 = base64.b64encode(original_buffer).decode('utf-8')
    compressed_base64 = base64.b64encode(compressed_buffer).decode('utf-8')
    
    # Clean up temporary files
    os.remove(temp_original)
    os.remove(temp_compressed)
    
    # Prepare result data
    result = {
        "original_image": original_base64,
        "compressed_image": compressed_base64,
        "original_size": f"{original_size:.2f}",
        "compressed_size": f"{compressed_size:.2f}",
        "compression_ratio": f"{original_size/compressed_size:.2f}",
        "method": f"{method} (Quality: {jpeg_quality})" if method == "JPEG" else f"{method} (Level: {png_level})",
        "psnr": f"{psnr:.2f}",
        "ssim": f"{ssim_value:.4f}",
        "comparison_plot": comparison_plot
    }
    
    return templates.TemplateResponse("compression.html", {
        "request": request,
        "result": result
    })

@app.post("/process_single/", response_class=HTMLResponse)
async def process_single_image(
    request: Request,
    image: UploadFile = File(...),
    operations: List[str] = Form(...),
    arithmetic_value: int = Form(50),
    threshold_value: int = Form(127),
    kernel_size: int = Form(3),
    padding_size: int = Form(10),
    kernel_type: str = Form(None),
    filter_type: str = Form(None)
):
    # Read the image
    image_data = await image.read()
    np_array = np.frombuffer(image_data, np.uint8)
    original_img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    results = {}
    results['original'] = save_image(original_img, "original")
    
    # Process each selected operation
    for operation in operations:
        if operation == "rgb_display":
            b, g, r = cv2.split(original_img)
            results['rgb_array'] = {
                "R": r.tolist(),
                "G": g.tolist(),
                "B": b.tolist()
            }
        
        elif operation == "grayscale":
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            results['grayscale'] = save_image(gray_img, "grayscale")
        
        elif operation == "inverse":
            inverse_img = cv2.bitwise_not(original_img)
            results['inverse'] = save_image(inverse_img, "inverse")
        
        elif operation == "histogram":
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            results['grayscale_histogram'] = save_histogram(gray_img, "grayscale")
            results['color_histogram'] = save_color_histogram(original_img)
        
        elif operation == "statistics":
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            results['statistics'] = {
                'mean': float(np.mean(gray_img)),
                'std': float(np.std(gray_img))
            }
        
        elif operation == "equalize":
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray_img)
            results['equalized'] = save_image(equalized, "equalized")
        
        elif operation == "canny_edge":
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_img, (kernel_size, kernel_size), 1.0)
            edges = cv2.Canny(blurred, 50, 150)
            results['edges'] = save_image(edges, "edges")
        
        elif operation == "chain_code":
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                chain_code = generate_freeman_chain_code(largest_contour)
                results['chain_code'] = chain_code
                
                # Draw contours
                img_contour = original_img.copy()
                cv2.drawContours(img_contour, [largest_contour], 0, (0, 255, 0), 2)
                results['contour_image'] = save_image(img_contour, "contour")
        
        elif operation == "integral_projection":
            gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)
            binary_norm = binary / 255.0
            h_proj, v_proj = calculate_projections(binary_norm)
            
            results['integral_projection'] = {
                'horizontal': h_proj.tolist(),
                'vertical': v_proj.tolist()
            }
        
        elif operation == "convolution":
            # Define kernel based on type
            if kernel_type == "average":
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
                kernel_name = "Average (Blur)"
            elif kernel_type == "sharpen":
                kernel = np.array([[-1,-1,-1],
                                 [-1, 9,-1],
                                 [-1,-1,-1]])
                kernel_name = "Sharpen"
            elif kernel_type == "edge":
                kernel = np.array([[-1,-1,-1],
                                 [-1, 8,-1],
                                 [-1,-1,-1]])
                kernel_name = "Edge Detection"
            
            # Apply convolution
            result = cv2.filter2D(original_img, -1, kernel)
            results['convolution'] = save_image(result, "convolution")
            results['convolution_type'] = kernel_name
            results['kernel'] = kernel.tolist()
        
        elif operation == "padding":
            # Apply zero padding
            padded = cv2.copyMakeBorder(
                original_img,
                padding_size, padding_size, padding_size, padding_size,
                cv2.BORDER_CONSTANT,
                value=[0,0,0]
            )
            results['padding'] = save_image(padded, "padded")
            results['padding_size'] = padding_size
        
        elif operation == "filter":
            if filter_type == "lowpass":
                # Gaussian blur for low pass filter
                result = cv2.GaussianBlur(original_img, (kernel_size, kernel_size), 0)
                filter_name = "Low Pass"
            elif filter_type == "highpass":
                # High pass filter with laplacian
                blur = cv2.GaussianBlur(original_img, (kernel_size, kernel_size), 0)
                result = cv2.addWeighted(original_img, 1.5, blur, -0.5, 0)
                filter_name = "High Pass"
            elif filter_type == "bandpass":
                # Band pass filter (kombinasi low dan high pass)
                blur1 = cv2.GaussianBlur(original_img, (kernel_size, kernel_size), 0)
                blur2 = cv2.GaussianBlur(original_img, (kernel_size*2, kernel_size*2), 0)
                result = cv2.subtract(blur1, blur2)
                filter_name = "Band Pass"
            
            results['filter'] = save_image(result, "filter")
            results['filter_type'] = filter_name
            results['filter_kernel_size'] = kernel_size
    
    return templates.TemplateResponse("compilation_result.html", {
        "request": request,
        "results": results,
        "operations": operations
    })

@app.post("/process_dual/", response_class=HTMLResponse)
async def process_dual_images(
    request: Request,
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    dual_operation: str = Form(...)
):
    # Read both images
    image1_data = await image1.read()
    image2_data = await image2.read()
    
    np_array1 = np.frombuffer(image1_data, np.uint8)
    np_array2 = np.frombuffer(image2_data, np.uint8)
    
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)
    
    results = {}
    results['original1'] = save_image(img1, "original1")
    results['original2'] = save_image(img2, "original2")
    
    if dual_operation == "and":
        result_img = cv2.bitwise_and(img1, img2)
        results['operation_result'] = save_image(result_img, "and_result")
        results['operation_name'] = "Bitwise AND"
    
    elif dual_operation == "xor":
        result_img = cv2.bitwise_xor(img1, img2)
        results['operation_result'] = save_image(result_img, "xor_result")
        results['operation_name'] = "Bitwise XOR"
    
    elif dual_operation == "specify":
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Simple histogram specification using histogram equalization
        equalized = cv2.equalizeHist(gray1)
        results['operation_result'] = save_image(equalized, "specified_result")
        results['operation_name'] = "Histogram Specification"
        
        # Add histograms
        results['hist_original'] = save_histogram(gray1, "hist_original")
        results['hist_reference'] = save_histogram(gray2, "hist_reference")
        results['hist_result'] = save_histogram(equalized, "hist_result")
    
    return templates.TemplateResponse("dual_result.html", {
        "request": request,
        "results": results,
        "operation": dual_operation
    })

def rgb_to_yiq(rgb):
    """
    Convert RGB to YIQ using transformation matrix.
    """
    transform_matrix = np.array([[0.299, 0.587, 0.114],
                               [0.59590059, -0.27455667, -0.32134392],
                               [0.21153661, -0.52273617, 0.31119955]])
    yiq = np.dot(rgb, transform_matrix.T)
    yiq = (yiq - np.min(yiq)) / (np.max(yiq) - np.min(yiq))  # Normalize
    return yiq

def plot_to_base64(fig):
    """
    Convert a matplotlib figure to base64 string.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()

def normalize_and_encode(image):
    """
    Normalize image and convert to base64.
    """
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    _, buffer = cv2.imencode('.png', normalized)
    return base64.b64encode(buffer).decode()

@app.get("/color_space/", response_class=HTMLResponse)
async def color_space_form(request: Request):
    return templates.TemplateResponse("color_space.html", {"request": request})

@app.post("/color_space/", response_class=HTMLResponse)
async def convert_color_space(
    request: Request,
    image: UploadFile = File(...),
    color_space: str = Form(...)
):
    # Read and process the image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image to selected color space and get components
    components = []
    converted_image = None

    if color_space == "xyz":
        converted_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2XYZ)
        X, Y, Z = cv2.split(converted_image)
        components = [
            {"title": "X Component", "image": normalize_and_encode(X)},
            {"title": "Y Component (Luminance)", "image": normalize_and_encode(Y)},
            {"title": "Z Component", "image": normalize_and_encode(Z)}
        ]

    elif color_space == "lab":
        converted_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        L, a, b = cv2.split(converted_image)
        components = [
            {"title": "L Component (Luminance)", "image": normalize_and_encode(L)},
            {"title": "a Component (Green-Red)", "image": normalize_and_encode(a)},
            {"title": "b Component (Blue-Yellow)", "image": normalize_and_encode(b)}
        ]

    elif color_space == "ycbcr":
        converted_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(converted_image)
        components = [
            {"title": "Y Component (Luminance)", "image": normalize_and_encode(Y)},
            {"title": "Cb Component (Blue Chrominance)", "image": normalize_and_encode(Cb)},
            {"title": "Cr Component (Red Chrominance)", "image": normalize_and_encode(Cr)}
        ]

    elif color_space == "yiq":
        converted_image = rgb_to_yiq(img_rgb / 255.0)
        Y = converted_image[:, :, 0]
        I = converted_image[:, :, 1]
        Q = converted_image[:, :, 2]
        components = [
            {"title": "Y Component (Luminance)", "image": normalize_and_encode((Y * 255).astype(np.uint8))},
            {"title": "I Component (In-phase)", "image": normalize_and_encode((I * 255).astype(np.uint8))},
            {"title": "Q Component (Quadrature)", "image": normalize_and_encode((Q * 255).astype(np.uint8))}
        ]

    elif color_space == "yuv":
        converted_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
        Y, U, V = cv2.split(converted_image)
        components = [
            {"title": "Y Component (Luminance)", "image": normalize_and_encode(Y)},
            {"title": "U Component", "image": normalize_and_encode(U)},
            {"title": "V Component", "image": normalize_and_encode(V)}
        ]

    # Convert original and converted images to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    original_image = base64.b64encode(buffer).decode()

    if color_space != "yiq":
        _, buffer = cv2.imencode('.png', converted_image)
        converted_image_base64 = base64.b64encode(buffer).decode()
    else:
        converted_image_base64 = normalize_and_encode((converted_image * 255).astype(np.uint8))

    return templates.TemplateResponse(
        "color_space.html",
        {
            "request": request,
            "original_image": original_image,
            "converted_image": converted_image_base64,
            "components": components
        }
    )

