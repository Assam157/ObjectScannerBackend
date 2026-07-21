import io
import os
import base64

from flask import Flask, request, jsonify, send_from_directory
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

print("YOLO Loaded Successfully")

###########################################################
# Serve React Frontend (NEW)
###########################################################

@app.route('/')
def serve_frontend():
    return send_from_directory('build', 'index.html')

@app.route('/<path:path>')
def serve_static_files(path):
    # If the requested file exists in the build folder, serve it
    if os.path.exists(os.path.join('build', path)):
        return send_from_directory('build', path)
    else:
        # For any other path (like React routes), serve index.html
        return send_from_directory('build', 'index.html')

###########################################################
# Detect API (UNCHANGED)
###########################################################

@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:

        return jsonify({

            "success":False,

            "message":"No image uploaded."

        }),400

    file = request.files["image"]

    if file.filename == "":

        return jsonify({

            "success":False,

            "message":"Empty filename."

        }),400

    ######################################################
    # Read image
    ######################################################

    image = Image.open(
        file.stream
    ).convert("RGB")

    ######################################################
    # YOLO
    ######################################################

    results = model.predict(

        image,

        conf=0.30,

        verbose=False

    )[0]

    ######################################################
    # Draw Boxes
    ######################################################

    draw = ImageDraw.Draw(image)

    detected_labels = []

    confidences = []

    highest_conf = 0

    detected_object = "none"

    for box in results.boxes:

        x1,y1,x2,y2 = box.xyxy[0].tolist()

        cls = int(box.cls[0])

        conf = float(box.conf[0])

        label = str(
            model.names[cls]
        ).lower()

        detected_labels.append(label)

        confidences.append(conf)

        if conf > highest_conf:

            highest_conf = conf

            detected_object = label

        draw.rectangle(

            [x1,y1,x2,y2],

            outline="red",

            width=4

        )

        draw.text(

            (x1,max(0,y1-20)),

            f"{label} {conf:.2f}",

            fill="red"

        )
            ######################################################
    # Convert image to Base64
    ######################################################

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG"
    )

    encoded_image = base64.b64encode(

        buffer.getvalue()

    ).decode()

    ######################################################
    # Decide Object
    ######################################################

    object_type = "notfound"

    if "chair" in detected_labels:

        object_type = "chair"

    elif (

        "monitor" in detected_labels

        or

        "tv" in detected_labels

        or

        "tvmonitor" in detected_labels

    ):

        object_type = "monitor"

    ######################################################
    # JSON Response
    ######################################################

    return jsonify({

        "success":True,

        "object":object_type,

        "detected":detected_object,

        "confidence":highest_conf,

        "image":encoded_image,

        "labels":detected_labels

    })

###########################################################
# Error Handlers
###########################################################

@app.errorhandler(404)
def page_not_found(e):

    return jsonify({

        "success":False,

        "message":"Endpoint not found."

    }),404


@app.errorhandler(500)
def internal_error(e):

    return jsonify({

        "success":False,

        "message":"Internal server error."

    }),500
###########################################################
# Main
###########################################################

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

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
