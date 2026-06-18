from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import docx

# ─── Load Rhamnus Model ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

rhamnus_model_path = os.path.join(BASE_DIR, "model", "rahamnus.h5")
rhamnus_model = load_model(rhamnus_model_path)
print('@@ Rhamnus Model loaded')

# Document files for each Rhamnus disease class
rhamnus_class_docs = {
    "Healthy": os.path.join(BASE_DIR, "Healthy Doc.docx"),
    "Powdery": os.path.join(BASE_DIR, "Powdery Doc.docx"),
    "Rust":    os.path.join(BASE_DIR, "Rust Doc.docx"),
    "error":   None
}


# ─── Prediction Function ──────────────────────────────────────────────────────
def pred_rhamnus(image_path):
    """Predict the disease class for a Rhamnus Davurica leaf image."""
    test_image = load_img(image_path, target_size=(225, 225))
    print("@@ Got image for prediction:", image_path)

    test_image = img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)

    result = rhamnus_model.predict(test_image).round(3)
    print('@@ Raw result =', result)

    max_idx = np.argmax(result)
    confidence = float(result[0, max_idx])

    if confidence >= 0.75:
        label_map = {0: "Healthy", 1: "Powdery", 2: "Rust"}
        label = label_map.get(max_idx, "error")
        return label, confidence, rhamnus_class_docs.get(label)
    else:
        return "Uncertain", confidence, None


# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')


@app.route("/analysis", methods=['GET'])
def analysis():
    return render_template('Analysis.html')


@app.route("/predict", methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file or file.filename == '':
        return redirect(url_for('analysis'))

    filename = file.filename
    print("@@ Input posted =", filename)

    upload_dir = os.path.join(BASE_DIR, 'static', 'user uploaded')
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)
    file.save(file_path)

    print("@@ Predicting class...")
    pred_label, confidence_score, doc_file = pred_rhamnus(file_path)

    return render_template(
        'predict.html',
        pred_output=pred_label,
        confidence_score=round(confidence_score * 100, 1),
        doc_file=doc_file,
        user_image=file_path.replace("\\", "/")
    )


@app.route("/download/<path:filename>", methods=['GET'])
def download(filename):
    if os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    return "File not found", 404


@app.route("/view_document", methods=['GET'])
def view_document():
    doc_file = request.args.get('doc_file')
    if doc_file and os.path.exists(doc_file):
        doc_content = read_document(doc_file)
        return render_template('view_document.html', doc_content=doc_content)
    return render_template('view_document.html', doc_content=["No document available."])


def read_document(doc_path):
    """Read and return paragraphs from a .docx file."""
    content = []
    try:
        doc = docx.Document(doc_path)
        for para in doc.paragraphs:
            if para.text.strip():
                content.append(para.text)
    except Exception as e:
        print(f"Error reading document: {e}")
        content = ["Could not read the document."]
    return content


@app.route("/feedback", methods=['GET'])
def feedback():
    return render_template('feedback.html')


@app.route("/submit_feedback", methods=['POST'])
def submit_feedback():
    return redirect(url_for('home'))


if __name__ == "__main__":
    app.run(debug=False, threaded=True)
