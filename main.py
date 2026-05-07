from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Check top-5 predictions: if ANY is in NOT_SKIN at >= this confidence → reject.
NOT_SKIN_THRESHOLD = 0.15  # Plus strict (était 0.20)

NOT_SKIN = {
    # Poultry / meat (cooked or raw)
    "hen", "cock", "rooster", "turkey", "duck", "goose",
    "ox", "hog", "pig", "boar",
    # Food items
    "cheeseburger", "hamburger", "hotdog", "hot_dog", "pizza",
    "spaghetti", "spaghetti_squash", "carbonara", "meatloaf", "meat_loaf",
    "potpie", "pot_pie", "pretzel", "bagel", "croissant", "french_loaf",
    "waffle", "pancake", "burrito", "taco", "sandwich",
    "ice_cream", "chocolate", "guacamole", "eggnog",
    "consomme", "hotpot", "stew", "broccoli", "cauliflower", "banana", "apple",
    # Containers / kitchen
    "plate", "bowl", "wok", "frying_pan", "ladle", "spatula",
    "pot", "cup", "bottle", "coffee_mug", "beaker",
    # Animals
    "dog", "cat", "fish", "snake", "insect", "spider",
    "hamster", "rabbit", "horse", "cow", "elephant", "bird",
    # Vehicles
    "car", "truck", "bus", "train", "airplane", "bicycle", "motorcycle", "van",
    "ambulance", "cab", "taxi", "sports_car", "jeep",
    # Objects / furniture
    "laptop", "keyboard", "remote_control", "phone", "television",
    "chair", "table", "sofa", "bed", "desk", "monitor", "mouse",
    "pillow", "curtain", "lamp", "bookshelf", "wardrobe",
    # Nature
    "flower", "mushroom", "tree", "leaf", "grass", "cloud", "mountain",
    # Clothing / Textile
    "t-shirt", "jean", "sweater", "sock", "shoe", "sneaker", "sandal",
    "velvet", "cloth", "fabric", "towel", "blanket", "carpet"
}

# If MobileNetV2 top-1 matches one of these → always accept (skin-compatible).
SKIN_LOOKALIKE = {
    "band_aid", "bandage", "lipstick", "sunscreen", "lotion", "gauze",
    "wig", "wool", "coral", "fur", "velvet", "sponge",
    "shower_cap", "diaper", "face_powder", "hand"
}

# Reject anything with top-1 confidence above this (clearly recognizable).
HIGH_CONF_THRESHOLD = 0.60  # Plus strict (était 0.70)
MIN_SKIN_CONFIDENCE = 45.0  # Seuil minimal pour le modèle de peau (en %)

interpreter = None
input_index = None
output_index = None
labels = []
imagenet_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global interpreter, input_index, output_index, labels, imagenet_model
    interpreter = tf.lite.Interpreter(model_path="./converted_model.tflite")
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    imagenet_model = tf.keras.applications.MobileNetV2(weights="imagenet")
    print("Models loaded")
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def is_skin_image(image: Image.Image) -> bool:
    img = image.resize((224, 224))
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(
        np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    )
    probs = imagenet_model.predict(arr, verbose=0)[0]
    top5 = tf.keras.applications.mobilenet_v2.decode_predictions(
        probs[np.newaxis], top=5)[0]

    print(f"DEBUG top5: {[(l, round(float(c), 2)) for _, l, c in top5]}")

    for _, label, conf in top5:
        label_clean = label.lower().replace(" ", "_")

        # Always accept if it looks like a skin-compatible class
        if any(s in label_clean for s in SKIN_LOOKALIKE):
            print(f"DEBUG: accepted — '{label}' is skin-compatible")
            return True

        # Reject if it's a known non-skin class at low-to-moderate confidence
        if float(conf) >= NOT_SKIN_THRESHOLD and any(f in label_clean for f in NOT_SKIN):
            print(f"DEBUG: rejected — '{label}' at {conf:.2f}")
            return False

    # Reject if top-1 is very confident about anything not caught above
    top1_conf = float(top5[0][2])
    if top1_conf >= HIGH_CONF_THRESHOLD:
        print(f"DEBUG: rejected — high conf {top1_conf:.2f} on '{top5[0][1]}'")
        return False

    return True


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Step 1: ImageNet Pre-filter
    if not is_skin_image(image):
        return {
            "prediction": "Unknown",
            "confidence": 0,
            "valid": False,
            "message": "Image not recognized. Please provide a clear photo of affected skin.",
            "top3": []
        }

    # Step 2: Skin Model Prediction
    arr = np.expand_dims(np.array(image.resize((224, 224)), dtype=np.uint8), axis=0)
    interpreter.set_tensor(input_index, arr)
    interpreter.invoke()
    preds_raw = interpreter.get_tensor(output_index)[0]

    out_details = interpreter.get_output_details()[0]
    if out_details["dtype"] == np.uint8:
        scale, zero_point = out_details["quantization"]
        preds = (preds_raw.astype(np.float32) - zero_point) * scale
    else:
        preds = preds_raw.astype(np.float32)

    top3 = sorted(enumerate(preds), key=lambda x: -x[1])[:3]
    top_confidence = round(float(top3[0][1]) * 100, 2)

    # Step 3: Confidence Threshold
    if top_confidence < MIN_SKIN_CONFIDENCE:
        return {
            "prediction": "Unknown",
            "confidence": top_confidence,
            "valid": False,
            "message": "Inconclusive result. Please ensure the photo is clear and well-lit.",
            "top3": [{"disease": labels[i], "confidence": round(float(p) * 100, 2)} for i, p in top3]
        }

    return {
        "prediction": labels[top3[0][0]],
        "confidence": top_confidence,
        "valid": True,
        "message": "Skin condition detected.",
        "top3": [{"disease": labels[i], "confidence": round(float(p) * 100, 2)} for i, p in top3]
    }
