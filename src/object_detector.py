#!/usr/bin/env python3
"""
Computer Vision Object Detection
Real-time object detection system using OpenCV and pre-trained models.
"""

import cv2  # type: ignore[import-not-found]
import numpy as np
import argparse
from flask import Flask, request, jsonify, render_template_string
import base64
import io
import os


class ObjectDetector:
    """Real-time object detection using YOLO and OpenCV."""

    def __init__(
        self,
        config_path="config/yolov3.cfg",
        weights_path="config/yolov3.weights",
        names_path="config/coco.names",
    ):
        self.net = None
        self.classes = []
        self.colors = []
        self.output_layers = []

        self.config_path = config_path
        self.weights_path = weights_path
        self.names_path = names_path

        self.initialize_detector()

    def initialize_detector(self):
        """Initialize the object detector with YOLOv3."""
        try:
            if not os.path.exists(self.weights_path):
                raise FileNotFoundError(
                    f"YOLOv3 weights file not found at {self.weights_path}. Please download it manually."
                )
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(
                    f"YOLOv3 config file not found at {self.config_path}. Please download it manually."
                )
            if not os.path.exists(self.names_path):
                raise FileNotFoundError(
                    f"COCO names file not found at {self.names_path}. Please download it manually."
                )

            # Load YOLO
            self.net = cv2.dnn.readNet(self.weights_path, self.config_path)

            # Use CUDA if available, otherwise fall back to CPU
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            with open(self.names_path, "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            layer_names = self.net.getLayerNames()
            unconnected = self.net.getUnconnectedOutLayers()
            if len(unconnected.shape) == 2:
                # OpenCV < 4.5.4: returns [[idx1], [idx2], ...]
                self.output_layers = [layer_names[i[0] - 1] for i in unconnected]
            else:
                # OpenCV >= 4.5.4: returns [idx1, idx2, ...]
                self.output_layers = [layer_names[i - 1] for i in unconnected]
            self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            print("YOLOv3 object detector initialized successfully!")
        except Exception as e:
            print(f"Error initializing YOLOv3 detector: {e}")
            print(
                "Please ensure yolov3.cfg, yolov3.weights, and coco.names are in the config/ directory."
            )
            self.net = None
            return

    def detect_objects(self, image):
        """Detect objects in image using YOLOv3."""
        if self.net is None:
            raise RuntimeError("Object detector network is not initialized")

        height, width, channels = image.shape

        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        # Showing information on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detections_list = []
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(image, label, (x, y + 30), font, 3, color, 3)

                detections_list.append(
                    {"class": label, "confidence": confidences[i], "bbox": [x, y, w, h]}
                )

        return image, detections_list

    def detect_from_webcam(self):
        """Real-time detection from webcam."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print('Press "q" to quit')

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect objects
            result_frame, detections = self.detect_objects(frame)

            # Display FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(
                result_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # Show frame
            cv2.imshow("Object Detection", result_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_from_image(self, image_path):
        """Detect objects in static image."""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image {image_path}")
            return None, []

        result_image, detections = self.detect_objects(image)
        return result_image, detections


# Flask Web Application
app = Flask(__name__)

# Lazy detector initialization
_detector = None


def get_detector():
    """Lazily initialize the detector when first needed."""
    global _detector
    if _detector is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        _detector = ObjectDetector(
            config_path=os.path.join(base_dir, "config", "yolov3.cfg"),
            weights_path=os.path.join(base_dir, "config", "yolov3.weights"),
            names_path=os.path.join(base_dir, "config", "coco.names"),
        )
    return _detector

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Computer Vision Object Detection</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .upload-area { border: 2px dashed #ddd; padding: 40px; text-align: center; margin: 20px 0; border-radius: 10px; }
        .upload-area:hover { border-color: #007bff; background: #f8f9fa; }
        input[type="file"] { display: none; }
        .upload-btn { background: #007bff; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        .upload-btn:hover { background: #0056b3; }
        .result { margin-top: 20px; }
        .image-container { text-align: center; margin: 20px 0; }
        .detected-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .detections { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 15px; }
        .detection-item { background: white; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 4px solid #007bff; }
        .webcam-btn { background: #28a745; color: white; padding: 12px 30px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }
        .webcam-btn:hover { background: #218838; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Computer Vision Object Detection</h1>
        
        <div class="upload-area" onclick="document.getElementById('imageInput').click()">
            <p>Click here to upload an image for object detection</p>
            <input type="file" id="imageInput" accept="image/*" onchange="uploadImage()">
            <button class="upload-btn">Choose Image</button>
        </div>
        
        <div style="text-align: center;">
            <button class="webcam-btn" onclick="startWebcam()">Start Webcam Detection</button>
            <p><small>Note: Webcam detection requires running the Python script locally</small></p>
        </div>
        
        <div id="result" class="result"></div>
    </div>

    <script>
        async function uploadImage() {
            const input = document.getElementById('imageInput');
            const file = input.files[0];
            
            if (!file) return;
            
            const formData = new FormData();
            formData.append('image', file);
            
            try {
                // This is a static demo. For actual detection, run the Flask app locally.
                // The API endpoint would be http://localhost:5000/detect
                alert("This is a static demo. Please run the Flask application locally to use the object detection functionality.");
                document.getElementById('result').innerHTML = '<p style="color: orange;">Static demo: Please run the Flask app locally for detection.</p>';
                
                // Example of how to call the API if the Flask app was running remotely:
                /*
                const response = await fetch('/detect', { // Or your remote API endpoint
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayResult(result);
                */

            } catch (error) {
                console.error('Error:', error);
                document.getElementById('result').innerHTML = '<p style="color: red;">Error processing image</p>';
            }
        }
        
        function displayResult(result) {
            const resultDiv = document.getElementById('result');
            
            let html = '<div class="image-container">';
            html += `<img src="data:image/jpeg;base64,${result.image}" class="detected-image" alt="Detected Objects">`;
            html += '</div>';
            
            if (result.detections && result.detections.length > 0) {
                html += '<div class="detections">';
                html += '<h3>Detected Objects:</h3>';
                
                result.detections.forEach(detection => {
                    html += `
                        <div class="detection-item">
                            <strong>${detection.class}</strong> - 
                            Confidence: ${(detection.confidence * 100).toFixed(1)}%<br>
                            <small>Location: (${detection.bbox[0]}, ${detection.bbox[1]}) 
                            Size: ${detection.bbox[2]}x${detection.bbox[3]}</small>
                        </div>
                    `;
                });
                
                html += '</div>';
            } else {
                html += '<div class="detections"><p>No objects detected in this image.</p></div>';
            }
            
            resultDiv.innerHTML = html;
        }
        
        function startWebcam() {
            alert('To use webcam detection, run the Python script locally with: python src/object_detector.py --webcam');
        }
    </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/detect", methods=["POST"])
def detect():
    """Detect objects in uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400

    try:
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect objects
        result_image, detections = get_detector().detect_objects(image)

        # Encode result image
        _, buffer = cv2.imencode(".jpg", result_image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"image": image_base64, "detections": detections})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Computer Vision Object Detection")
    parser.add_argument(
        "--webcam", action="store_true", help="Use webcam for real-time detection"
    )
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--web", action="store_true", help="Start web server")

    args = parser.parse_args()

    print("Computer Vision Object Detection System")
    print("=" * 40)

    if args.webcam:
        print("Starting webcam detection...")
        get_detector().detect_from_webcam()
    elif args.image:
        print(f"Processing image: {args.image}")
        result_image, detections = get_detector().detect_from_image(args.image)
        if result_image is not None:
            print(f"Found {len(detections)} objects:")
            for detection in detections:
                print(f"  - {detection['class']}: {detection['confidence']:.2f}")

            # Save result
            output_path = os.path.join(os.path.dirname(args.image), f"detected_{os.path.basename(args.image)}")
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
    else:
        print("Starting web server...")
        print("Open http://localhost:5000 in your browser")
        app.run(debug=False, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
