from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import docx

# Load model for Mango
mango_model_path = "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB 2/model/mango.h5"
mango_model = load_model(mango_model_path)
print('@@ Mango Model loaded')

# Load model for Rahamnus
rahamnus_model_path = "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB 2/model/rahamnus.h5"
rahamnus_model = load_model(rahamnus_model_path)
print('@@ Rahamnus Model loaded')

# Define a dictionary mapping class labels to document files for Mango
mango_class_docs = {
    "Alternaria Leaf Spot": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB 2/Alternaria Leaf Spot Doc.docx",
    "Black Rot": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB 2/Black Rot Doc.docx",
    "Brown Spot": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB 2/Brown Spot Doc.docx",
    "Healthy": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB 2/Healthy Doc.docx",
    "Rust": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB 2/Rust.docx",
    "Scab": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB 2/Scab Doc.docx",
    "error": None
}

# Define a dictionary mapping class labels to document files for Rahamnus
rahamnus_class_docs = {
    "Healthy": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB/Healthy Doc.docx",
    "Powdery": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB/Powdery Doc.docx",
    "Rust": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB/Rust Doc.docx",
    "error": None
}


# Function to predict the class for Mango
def pred_the_detection_for_mango(detection):
    test_image = load_img(detection, target_size=(225, 225))  # Load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # Convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimension 3D to 4D

    result = mango_model.predict(test_image).round(6)  # Predict class
    print('@@ Raw result = ', result)

    max_confidence_index = np.argmax(result)  # Get the index of max confidence value
    max_confidence = result[0, max_confidence_index]  # Get the max confidence score

    if max_confidence >= 0.60:
        if max_confidence_index == 0:
            return "Alternaria Leaf Spot", max_confidence, mango_class_docs["Alternaria Leaf Spot"]  # If index 0
        elif max_confidence_index == 1:
            return "Black Rot", max_confidence, mango_class_docs["Black Rot"]  # If index 1
        elif max_confidence_index == 2:
            return "Brown Spot", max_confidence, mango_class_docs["Brown Spot"]  # If index 2
        elif max_confidence_index == 3:
            return "Healthy", max_confidence, mango_class_docs["Healthy"]  # If index 3
        elif max_confidence_index == 4:
            return "Rust", max_confidence, mango_class_docs["Rust"]  # If index 4
        elif max_confidence_index == 5:
            return "Scab", max_confidence, mango_class_docs["Scab"]  # If index 5
        else:
            return "error", 0, mango_class_docs["error"]
    else:
        return "Uncertain", max_confidence, None


# Function to predict the class for Rahamnus
def pred_the_detection_for_rahamnus(detection):
    test_image = load_img(detection, target_size=(225, 225))  # Load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # Convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimension 3D to 4D

    result = rahamnus_model.predict(test_image).round(3)  # Predict class
    print('@@ Raw result = ', result)

    max_confidence_index = np.argmax(result)  # Get the index of max confidence value
    max_confidence = result[0, max_confidence_index]  # Get the max confidence score

    if max_confidence >= 0.75:
        if max_confidence_index == 0:
            return "Healthy", max_confidence, rahamnus_class_docs["Healthy"]  # If index 0
        elif max_confidence_index == 1:
            return "Powdery", max_confidence, rahamnus_class_docs["Powdery"]  # If index 1
        elif max_confidence_index == 2:
            return "Rust", max_confidence, rahamnus_class_docs["Rust"]  # If index 2
        else:
            return "error", 0, rahamnus_class_docs["error"]
    else:
        return "Uncertain", max_confidence, None

# Create Flask instance
app = Flask(__name__)


# render home.html page
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')


# render analysis.html page
@app.route("/analysis", methods=['GET', 'POST'])
def analysis():
    return render_template('analysis.html')


# Route to predict the class and provide options for viewing or downloading document
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # Get input
        filename = file.filename
        print("@@ Input posted = ", filename)

        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")

        # Example: Check if mango model should be used
        if request.form['crop'] == 'Mango':
            pred_label, confidence_score, doc_file = pred_the_detection_for_mango(detection=file_path)
        else:  # Otherwise, use Rahamnus model
            pred_label, confidence_score, doc_file = pred_the_detection_for_rahamnus(detection=file_path)

        return render_template('predict.html', pred_output=pred_label, confidence_score=confidence_score, doc_file=doc_file, user_image=file_path)


# Route to download the document file
@app.route("/download/<path:filename>", methods=['GET', 'POST'])
def download(filename):
    doc_path = os.path.join('static', 'user uploaded', filename)
    if os.path.exists(doc_path):
        return send_file(doc_path, as_attachment=True)
    else:
        return "File not found"

# Route to view document in a new tab or show its insights
@app.route("/view_document", methods=['GET'])
def view_document():
    doc_file = request.args.get('doc_file')
    if doc_file and os.path.exists(os.path.join('static', 'user uploaded', doc_file)):
        doc_content = read_document(doc_file)
        return render_template('view_document.html', doc_content=doc_content)
    else:
        insights = "No insights available"
        return render_template('insights.html', insights=insights)

# Function to read document content
def read_document(doc_file):
    doc_path = os.path.join('static', 'user uploaded', doc_file)
    doc_content = []
    try:
        doc = docx.Document(doc_path)
        for para in doc.paragraphs:
            doc_content.append(para.text)
    except Exception as e:
        print(f"Error reading document: {e}")
    return doc_content

# Render feedback.html page
@app.route("/feedback", methods=['GET', 'POST'])
def feedback():
    return render_template('feedback.html')

# Handle feedback submission
@app.route("/submit_feedback", methods=['POST'])
def submit_feedback():
    # Handle feedback submission here
    # Access form data using request.form dictionary
    # Process the feedback data as needed
    return redirect(url_for('home'))  # Redirect to home page after submission

if __name__ == "__main__":
    app.run(threaded=True)
