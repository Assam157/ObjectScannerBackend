import io
import os
import base64
import time

import numpy as np
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS

from PIL import Image, ImageDraw

from ultralytics import YOLO

###########################################################
# Flask
###########################################################

app = Flask(__name__)
CORS(app)

###########################################################
# Load YOLO
###########################################################

print("=" * 60)
print("Loading YOLO Model...")
print("=" * 60)

MODEL_PATH = "best (2).pt"

model = YOLO(MODEL_PATH)

# Warm up model (prevents slow first request)
dummy = np.zeros((640, 640, 3), dtype=np.uint8)

with torch.inference_mode():
    model.predict(
        dummy,
        imgsz=640,
        verbose=False
    )

print("YOLO Loaded Successfully")

###########################################################
# Health Check
###########################################################

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "message": "Object Scanner Backend"
    })

###########################################################
# Detect API
###########################################################

@app.route("/detect", methods=["POST"])
def detect():

    try:

        if "image" not in request.files:
            return jsonify({
                "success": False,
                "message": "No image uploaded."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "success": False,
                "message": "Empty filename."
            }), 400

        ##################################################
        # Read image
        ##################################################

        image = Image.open(file.stream).convert("RGB")

        # Reduce memory usage
        image.thumbnail((640, 640))

        ##################################################
        # Prediction
        ##################################################

        start = time.time()

        with torch.inference_mode():

            results = model.predict(
                image,
                imgsz=640,
                conf=0.30,
                verbose=False,
                device="cpu"
            )[0]

        print(f"Inference Time: {time.time() - start:.2f} sec")

        ##################################################
        # Draw boxes
        ##################################################

        draw = ImageDraw.Draw(image)

        detected_labels = []
        highest_conf = 0
        detected_object = "none"

        for box in results.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            label = model.names[cls].lower()

            detected_labels.append(label)

            if conf > highest_conf:
                highest_conf = conf
                detected_object = label

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 20)),
                f"{label} {conf:.2f}",
                fill="red"
            )

        ##################################################
        # Convert image to Base64
        ##################################################

        buffer = io.BytesIO()

        image.save(buffer, format="JPEG")

        encoded_image = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        ##################################################
        # Decide object type
        ##################################################

        object_type = "notfound"

        if "chair" in detected_labels:
            object_type = "chair"

        elif (
            "monitor" in detected_labels
            or "tv" in detected_labels
            or "tvmonitor" in detected_labels
        ):
            object_type = "monitor"

        ##################################################
        # Response
        ##################################################

        return jsonify({
            "success": True,
            "object": object_type,
            "detected": detected_object,
            "confidence": highest_conf,
            "labels": detected_labels,
            "image": encoded_image
        })

    except Exception as e:

        print("ERROR:", str(e))

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


###########################################################
# Error Handlers
###########################################################

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({
        "success": False,
        "message": "Endpoint not found."
    }), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "success": False,
        "message": "Internal server error."
    }), 500


###########################################################
# Main
###########################################################

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("=" * 60)
    print("Object Scanner Backend Started")
    print(f"Listening on Port {port}")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=True
    )
