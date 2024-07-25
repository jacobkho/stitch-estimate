from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import shutil

app = Flask(__name__)
TEMP_DIR = 'static/temp/'

# Ensure the temporary directory exists
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.route('/', methods=['GET', 'POST'])
def index():
    error = None
    cleaned_filepath = None
    area = 0
    stitches = 0
    cost = 0

    if request.method == 'POST':
        # Clean up the temp directory
        shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR)
        
        file = request.files['file']
        desired_length = request.form.get('length')
        desired_width = request.form.get('width')

        desired_length = float(desired_length) if desired_length else 0
        desired_width = float(desired_width) if desired_width else 0

        if file:
            filepath = os.path.join(TEMP_DIR, file.filename)
            file.save(filepath)
            print(f"File uploaded to {filepath}")

            # Check the file format
            file_format = file.content_type
            print(f"Uploaded file format: {file_format}")
            supported_formats = ["image/jpeg", "image/png"]
            if file_format not in supported_formats:
                print("Unsupported file format")
                error = "Unsupported file format"
            else:
                area, stitches, cost = calculate_estimates(filepath, desired_length, desired_width)
                cleaned_filepath = filepath

    return render_template('index.html', filepath=cleaned_filepath, area=area, stitches=stitches, cost=cost, error=error)

def calculate_estimates(image_path, desired_length, desired_width):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to read the image")
        return 0, 0, 0

    print(f"Image read successfully: {image.shape}")

    # Extract DPI
    pil_image = Image.open(image_path)
    dpi = pil_image.info.get('dpi', (300, 300))
    dpi_x, dpi_y = dpi

    # Use default DPI if the extracted DPI is unusually low
    if dpi_x < 100 or dpi_y < 100:
        dpi_x, dpi_y = 300, 300
    print(f"DPI: {dpi_x}, {dpi_y}")

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image_path = os.path.join(TEMP_DIR, 'gray_image.png')
    cv2.imwrite(gray_image_path, gray_image)
    print(f"Grayscale image saved to {gray_image_path}")

    # Apply Otsu's threshold to convert the image to binary
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_image_path = os.path.join(TEMP_DIR, 'binary_image.png')
    cv2.imwrite(binary_image_path, binary_image)
    print(f"Binary image saved to {binary_image_path}")

    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours found")
        return 0, 0, 0

    object_contour = max(contours, key=cv2.contourArea)
    area_pixels = cv2.contourArea(object_contour)
    print(f"Contour area in pixels: {area_pixels}")

    # Draw contours on the image for visualization
    contour_image = cv2.drawContours(image.copy(), [object_contour], -1, (0, 255, 0), 3)
    contour_image_path = os.path.join(TEMP_DIR, 'contour_image.png')
    cv2.imwrite(contour_image_path, contour_image)
    print(f"Contour image saved to {contour_image_path}")

    if area_pixels == 0:
        return 0, 0, 0

    # Calculate the object's current width and length in inches
    contour_rect = cv2.minAreaRect(object_contour)
    box = cv2.boxPoints(contour_rect)
    box = np.int_(box)

    width_pixels = np.linalg.norm(box[0] - box[1])
    length_pixels = np.linalg.norm(box[1] - box[2])

    current_width_in_inches = width_pixels / dpi_x
    current_length_in_inches = length_pixels / dpi_y

    print(f"Current width in inches: {current_width_in_inches}, Current length in inches: {current_length_in_inches}")

    # Adjust dimensions based on the desired length or width
    scale_factor_length = scale_factor_width = 1
    if desired_length > 0 and desired_width > 0:
        scale_factor_length = desired_length / current_length_in_inches
        scale_factor_width = desired_width / current_width_in_inches
        scale_factor = (scale_factor_length + scale_factor_width) / 2
        adjusted_width_in_inches = current_width_in_inches * scale_factor
        adjusted_length_in_inches = current_length_in_inches * scale_factor
    elif desired_length > 0:
        scale_factor_length = desired_length / current_length_in_inches
        adjusted_length_in_inches = desired_length
        adjusted_width_in_inches = current_width_in_inches * scale_factor_length
    elif desired_width > 0:
        scale_factor_width = desired_width / current_width_in_inches
        adjusted_width_in_inches = desired_width
        adjusted_length_in_inches = current_length_in_inches * scale_factor_width
    else:
        adjusted_length_in_inches = current_length_in_inches
        adjusted_width_in_inches = current_width_in_inches

    print(f"Adjusted width in inches: {adjusted_width_in_inches}, Adjusted length in inches: {adjusted_length_in_inches}")

    adjusted_area_square_inches = round(adjusted_width_in_inches * adjusted_length_in_inches, 1)

    total_stitches = adjusted_area_square_inches * 2000
    total_cost = (total_stitches / 1000) * 1.25

    print(f"Adjusted area in square inches: {adjusted_area_square_inches}")
    print(f"Total stitches: {total_stitches}")
    print(f"Total cost: {total_cost}")

    return adjusted_area_square_inches, total_stitches, total_cost

@app.route('/file/<path:filename>')
def serve_file(filename):
    return send_file(os.path.join(TEMP_DIR, filename))

if __name__ == '__main__':
    app.run(debug=True)
