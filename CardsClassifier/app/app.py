from flask import Flask, request, jsonify
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import timm
import os


# Define Flask app
app = Flask(__name__)

# Class labels
class_labels = {
    0: 'ace of clubs', 1: 'ace of diamonds', 2: 'ace of hearts', 3: 'ace of spades',
    4: 'eight of clubs', 5: 'eight of diamonds', 6: 'eight of hearts', 7: 'eight of spades',
    8: 'five of clubs', 9: 'five of diamonds', 10: 'five of hearts', 11: 'five of spades',
    12: 'four of clubs', 13: 'four of diamonds', 14: 'four of hearts', 15: 'four of spades',
    16: 'jack of clubs', 17: 'jack of diamonds', 18: 'jack of hearts', 19: 'jack of spades',
    20: 'joker', 21: 'king of clubs', 22: 'king of diamonds', 23: 'king of hearts', 24: 'king of spades',
    25: 'nine of clubs', 26: 'nine of diamonds', 27: 'nine of hearts', 28: 'nine of spades',
    29: 'queen of clubs', 30: 'queen of diamonds', 31: 'queen of hearts', 32: 'queen of spades',
    33: 'seven of clubs', 34: 'seven of diamonds', 35: 'seven of hearts', 36: 'seven of spades',
    37: 'six of clubs', 38: 'six of diamonds', 39: 'six of hearts', 40: 'six of spades',
    41: 'ten of clubs', 42: 'ten of diamonds', 43: 'ten of hearts', 44: 'ten of spades',
    45: 'three of clubs', 46: 'three of diamonds', 47: 'three of hearts', 48: 'three of spades',
    49: 'two of clubs', 50: 'two of diamonds', 51: 'two of hearts', 52: 'two of spades'
}
class_names = list(class_labels.values())

# Preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Define models
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children()))[:-1]

        enet_out_size = 1280

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enet_out_size, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output

class InceptionResNetV2CardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(InceptionResNetV2CardClassifier, self).__init__()
        self.base_model = timm.create_model('inception_resnet_v2', pretrained=True)
        self.base_model.classif = nn.Linear(self.base_model.classif.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Helper functions
def preprocess_image(image_file, transform):
    try:
        image = Image.open(image_file).convert("RGB")
        return transform(image).unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy().flatten()
        top_prob, top_classes = torch.topk(torch.tensor(probabilities), 3)
        top_prob = top_prob.numpy()
        top_classes = top_classes.numpy()
        predictions = [{"class": class_names[idx], "probability": round(float(prob), 3)}
                        for idx, prob in zip(top_classes, top_prob)]
        return predictions

efficientnet_model_path = "card_classifier_enet.pth"
inception_model_path = "card_classifier_inception_resnet_v2.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load models and weights
efficientnet_model = SimpleCardClassifier()
inception_model = InceptionResNetV2CardClassifier()

# Ensure the model files exist before loading them
if not os.path.exists(efficientnet_model_path):
    raise FileNotFoundError(f"EfficientNet model file not found: {efficientnet_model_path}")
else:
    efficientnet_model.load_state_dict(torch.load(efficientnet_model_path, map_location=device))
    efficientnet_model.to(device)

if not os.path.exists(inception_model_path):
    raise FileNotFoundError(f"Inception model file not found: {inception_model_path}")
else:
    inception_model.load_state_dict(torch.load(inception_model_path, map_location=device))
    inception_model.to(device)

# Flask endpoints
@app.route('/predict', methods=['POST'])
def predict_card():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Preprocess the image for both models
        image_tensor_enet = preprocess_image(file, transform)  # For EfficientNet
        image_tensor_inception = preprocess_image(file, transform2)  # For InceptionResNetV2

        # Perform predictions with both models
        enet_predictions = predict(efficientnet_model, image_tensor_enet, device)
        inception_predictions = predict(inception_model, image_tensor_inception, device)

        return jsonify({
            "efficientnet_predictions": enet_predictions,
            "inception_predictions": inception_predictions
        })
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/hello', methods=['GET'])
def hello():
    return jsonify({'hello': 'ram'})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5002)
