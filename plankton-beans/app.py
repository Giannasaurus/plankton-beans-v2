import os

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

# ---- config ----
UPLOAD_FOLDER = "static/uploads"
IMG_SIZE = (160, 160)
# Per-class labels (index -> subcategory). Adjust order if your model uses a different class ordering.
CATEGORIES = [
    "battery",
    "biological",
    "trash",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes"
]

# Map numeric index -> human label
INDEX_TO_LABEL = {i: name for i, name in enumerate(CATEGORIES)}

# Which subcategories are considered recyclable
RECYCLABLE_CATEGORIES = {"cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- app ----
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---- load model ONCE ----
try:
    model = tf.keras.models.load_model("trained_models/best_model.keras", compile=False)
except Exception as e:
    print(f"Warning: failed to load model: {e}")
    model = None


# ---- utils ----s
def preprocess_image(img_path):
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
    except UnidentifiedImageError:
        raise ValueError("Uploaded file is not a valid image.")


# ---- routes ----
@app.route("/", methods=["GET", "POST"])
def index():
    category = None
    recyclable_label = None
    confidence_display = None
    is_recyclable = None
    class_probs = {}

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            try:
                img_array = preprocess_image(filepath)
            except ValueError:
                return render_template(
                    "index.html",
                    category="No image yet",
                    recyclable_label="—",
                    confidence="—",
                    is_recyclable=False,
                )

            if model is None:
                category = "Model not loaded"
                confidence_display = 0.0
                is_recyclable = False
            else:
                preds = model.predict(img_array)[0]
                preds = np.asarray(preds).flatten()
                n_preds = preds.shape[0]
                n_cats = len(CATEGORIES)

                # Build per-category probs for available outputs; fill missing with 0.0
                class_probs = {}
                for i in range(min(n_preds, n_cats)):
                    class_probs[CATEGORIES[i]] = float(preds[i])
                if n_preds < n_cats:
                    for i in range(n_preds, n_cats):
                        class_probs[CATEGORIES[i]] = 0.0
                # If model outputs more classes than we have labels, expose them as extra_class_i
                if n_preds > n_cats:
                    for i in range(n_cats, n_preds):
                        class_probs[f"extra_class_{i}"] = float(preds[i])

                # Aggregate recyclable probability using only indices that map to known categories
                recyclable_prob = float(
                    sum(preds[i] for i, c in enumerate(CATEGORIES) if i < n_preds and c in RECYCLABLE_CATEGORIES)
                )

                # Determine non-recyclable probability from known-category mass
                known_mass = float(sum(preds[i] for i in range(min(n_preds, n_cats))))
                nonrecyclable_prob = float(max(0.0, known_mass - recyclable_prob))

                top_level_probs = {"Recyclable": recyclable_prob, "Non-Recyclable": nonrecyclable_prob}

                # Pick top-level label and confidence
                if recyclable_prob >= nonrecyclable_prob:
                    category = "Recyclable"
                    confidence_display = round(recyclable_prob, 3)
                    is_recyclable = True
                else:
                    category = "Non-Recyclable"
                    confidence_display = round(nonrecyclable_prob, 3)
                    is_recyclable = False

                # Also record predicted subcategory for transparency (handle extra outputs)
                class_idx = int(np.argmax(preds))
                if class_idx < n_cats:
                    subcategory = INDEX_TO_LABEL.get(class_idx, "Unknown")
                else:
                    subcategory = f"extra_class_{class_idx}"
                class_probs["predicted_subcategory"] = subcategory
                class_probs["top_level_probs"] = top_level_probs

            recyclable_label = "Recyclable" if is_recyclable else "Non-Recyclable"

    return render_template(
    "index.html",
    category=category,
    recyclable_label=recyclable_label,
    confidence=confidence_display,
    is_recyclable=is_recyclable,
    class_probs=class_probs
)


if __name__ == "__main__":
    app.run(debug=True)
