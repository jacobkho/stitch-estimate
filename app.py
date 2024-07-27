from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Function to process the image and remove the background using GrabCut
def remove_background(image):
    # Convert image to numpy array
    image_np = np.array(image)

    # Create a mask
    mask = np.zeros(image_np.shape[:2], np.uint8)

    # Create temporary arrays used by GrabCut
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define the rectangle which contains the foreground object
    rect = (10, 10, image_np.shape[1]-10, image_np.shape[0]-10)

    # Apply GrabCut algorithm
    cv2.grabCut(image_np, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask so that sure and likely foreground are set to 1, and the rest are set to 0
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    result_image = image_np * mask2[:, :, np.newaxis]

    # Convert masked areas to white background
    background = np.full_like(result_image, (255, 255, 255))
    result_with_white_bg = np.where(result_image == 0, background, result_image)

    return result_with_white_bg, mask2

# Function to resize the image based on the user's specifications
def resize_image(image, mask, width_inch, height_inch, dpi):
    original_width, original_height = image.shape[1], image.shape[0]
    if width_inch and not height_inch:
        width_px = int(width_inch * dpi)
        height_px = int((width_px / original_width) * original_height)
    elif height_inch and not width_inch:
        height_px = int(height_inch * dpi)
        width_px = int((height_px / original_height) * original_width)
    else:
        width_px = int(width_inch * dpi)
        height_px = int(height_inch * dpi)

    resized_image = cv2.resize(image, (width_px, height_px), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(mask, (width_px, height_px), interpolation=cv2.INTER_AREA)
    return resized_image, resized_mask

# Function to calculate the area of the object in square inches and total stitches
def calculate_area_and_stitches(mask, dpi):
    object_area_pixels = np.sum(mask == 1)
    square_inches = object_area_pixels / (dpi * dpi)
    total_stitches = square_inches * 2000
    return square_inches, total_stitches

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    file = request.files['image']
    width_inch = request.form.get('width', type=float)
    height_inch = request.form.get('height', type=float)
    
    image = Image.open(file.stream).convert('RGB')
    dpi = image.info.get('dpi', (300, 300))[0]  # Default to 300 DPI if not found
    result_image, mask = remove_background(image)
    resized_image, resized_mask = resize_image(result_image, mask, width_inch, height_inch, dpi)
    square_inches, total_stitches = calculate_area_and_stitches(resized_mask, dpi)

    return jsonify({
        'square_inches': round(square_inches, 2),
        'total_stitches': round(total_stitches)
    })

if __name__ == "__main__":
    app.run(debug=True)
