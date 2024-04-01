from flask import Flask, render_template, request, send_file, redirect, url_for
import os
import numpy as np
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from PIL import Image
import docx

# Load model
model = load_model("B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB/model/best_model.h5")
print('@@ Model loaded')

# Define a dictionary mapping class labels to document files
class_docs = {
    "Healthy": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB/Healthy Doc.docx",
    "Powdery": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB/Powdery Doc.docx",
    "Rust": "B:/VESP/6th Sem/CPE/CAD WEB/CAD WEB/Rust Doc.docx",
    "error": None
}

# Function to predict the class
def pred_the_detection(detection):
    test_image = load_img(detection, target_size=(225, 225))  # Load image
    print("@@ Got Image for prediction")

    test_image = img_to_array(test_image) / 255  # Convert image to np array and normalize
    test_image = np.expand_dims(test_image, axis=0)  # Change dimension 3D to 4D

    result = model.predict(test_image).round(3)  # Predict class
    print('@@ Raw result = ', result)

    pred = np.argmax(result)  # Get the index of max value

    if pred == 0:
        return "Healthy", class_docs["Healthy"]  # If index 0
    elif pred == 1:
        return "Powdery", class_docs["Powdery"]  # If index 1
    elif pred == 2:
        return "Rust", class_docs["Rust"]  # If index 2
    else:
        return "error", class_docs["error"]

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
        pred_label, doc_file = pred_the_detection(detection=file_path)

        return render_template('predict.html', pred_output=pred_label, doc_file=doc_file, user_image=file_path)

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
