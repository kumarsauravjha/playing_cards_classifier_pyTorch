from flask import Flask, request, jsonify
from torchvision import models, transforms
from PIL import Image
import torch
import os

# Define Flask app
app = Flask(__name__)

# Load pre-trained VGG16 model
model = models.vgg16(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Save the model for demonstration purposes
MODEL_PATH = "vgg16.pth"
if not os.path.exists(MODEL_PATH):
    torch.save(model.state_dict(), MODEL_PATH)

# Reload the model
loaded_model = models.vgg16(pretrained=False)  # Load an untrained model architecture
loaded_model.load_state_dict(torch.load(MODEL_PATH))
loaded_model.eval()

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'hello' : 'ram'})

@app.route('/predict', methods=['POST'])
def predict():
    print("Request method:", request.method)
    print("Headers:", request.headers)
    print("Content-Type:", request.content_type)
    print("Form data:", request.form)
    print("Files:", request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Load the image from the file object
        img = Image.open(file).convert('RGB')  # Convert to RGB to ensure compatibility
        img_tensor = preprocess(img)  # Preprocess the image
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = loaded_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        # Decode top-3 predictions
        top3_prob, top3_classes = torch.topk(probabilities, 3)
        top3_prob = top3_prob.numpy()
        top3_classes = top3_classes.numpy()

        # Map class indices to human-readable labels
        with open("/app/imagenet_classes.txt") as f:
            labels = [line.strip() for line in f.readlines()]
        predictions = [{"class": labels[idx], "probability": float(prob)} for idx, prob in zip(top3_classes, top3_prob)]

        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)
