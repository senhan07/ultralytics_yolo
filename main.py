from PIL import Image
from ultralytics import YOLO
import os

# Load a pretrained YOLOv8 model
model = YOLO('yolov8x-seg.pt')

# Input and output directories
input_dir = "input"
output_dir = "output"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over each image in the input directory
for filename in os.listdir(input_dir):
    # Load the input image
    image_path = os.path.join(input_dir, filename)
    original_image = Image.open(image_path)

    # Run inference on the input image
    results = model(original_image)

    # Iterate over each result
    for i, r in enumerate(results):
        # Generate output filename
        output_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save result image to the output directory
        r.save(filename=output_path)

        print(f"Saved result to: {output_path}")
