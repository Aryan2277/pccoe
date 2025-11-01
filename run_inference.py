import cv2
import sys
import numpy as np
from flask import Flask, request, render_template_string, send_file
from inference import get_roboflow_model
import supervision as sv
import io # Used to handle image data in memory

# --- Initialize the Flask App ---
app = Flask(__name__)

# --- IMPORTANT: REPLACE THESE VALUES ---
# Make sure to use your RESET API key and FULL model ID
YOUR_API_KEY = "i3HLHjjhvWCpehLbgiSW" # Replace with your key
YOUR_MODEL_ID = "plastic-image-detection-ivewv/1" # Replace with your model ID

# --- Load your Roboflow model ---
try:
    model = get_roboflow_model(model_id=YOUR_MODEL_ID, api_key=YOUR_API_KEY)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check your API key and Model ID in app.py")
    sys.exit(1)
    
# --- Define the HTML for the frontend ---
# This simple HTML page has a form for uploading an image.
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Object Detection</title>
    <style>
        body { font-family: sans-serif; text-align: center; margin-top: 50px; }
        img { max-width: 80%; height: auto; margin-top: 20px; border: 2px solid #ccc; }
    </style>
</head>
<body>
    <h1>Upload an Image for Object Detection</h1>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <button type="submit">Detect Objects</button>
    </form>
    {% if image_data %}
        <h2>Result:</h2>
        <img src="data:image/jpeg;base64,{{ image_data }}">
    {% endif %}
</body>
</html>
"""

# --- Define the main route for the web application ---
@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Check if a file was uploaded
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        if file:
            # Read the image file from the request into memory
            image_bytes = file.read()
            # Convert the bytes to a NumPy array that OpenCV can use
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # --- Your original inference logic ---
            results = model.infer(image)[0]
            detections = sv.Detections.from_inference(results)

            # Annotate the image
            bounding_box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()
            
            annotated_image = bounding_box_annotator.annotate(
                scene=image.copy(), detections=detections
            )
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections
            )
            # --- End of inference logic ---
            
            # Instead of displaying, save the annotated image to a memory buffer
            is_success, buffer = cv2.imencode(".jpg", annotated_image)
            if not is_success:
                return "Error encoding image", 500
            
            # Encode the image buffer to base64 to embed it in the HTML
            import base64
            image_data = base64.b64encode(buffer).decode('utf-8')
            
            # Render the same page, but now with the result image
            return render_template_string(HTML_TEMPLATE, image_data=image_data)

    # For a GET request, just show the upload form
    return render_template_string(HTML_TEMPLATE, image_data=None)

# --- Run the Flask app ---
if __name__ == "__main__":
    app.run(debug=True)