#%%
import pandas as pandas
import torch
from torchvision import models
import timm
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
#%%
device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#%%
# Define the class labels as a dictionary
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

#%%
class_names = list(class_labels.values())
#%%
enet_model_path = "card_classifier_enet.pth"
inception_model_path = "card_classifier_inception_resnet_v2.pth"
nasnet_model_path = "card_classifier_nasnet.pth"
#%%
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

#%%
enet_model = SimpleCardClassifier()

enet_model.load_state_dict(torch.load(enet_model_path,weights_only=True, map_location=device))

#%%
enet_model.to(device)
#%%
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

#%%
transform2 = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
# %%
# Predict using the model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

#%%
# Load and preprocess the image
def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

#%%
def preprocess_image2(image_path, transform2):
    image = Image.open(image_path).convert("RGB")
    return image, transform2(image).unsqueeze(0)
#%%
def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    
    # Display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    
    # Display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()


# %%
test_image = "D:/STUDY/MS/DATS 6450 Computer Vision/playing_cards_classifier_pyTorch/playing card dataset/test/nine of hearts/4.jpg"

original_image, image_tensor = preprocess_image(test_image, transform)
prob = predict(enet_model, image_tensor, device)
# %%
print(prob)

#%%
# Find the index of the highest probability
predicted_index = prob.argmax()  # or use torch.argmax(prob) if prob is a tensor

# Map the predicted index to the card label
predicted_card = class_labels[predicted_index.item()]

print(f"The predicted card is: {predicted_card}")

# %%
visualize_predictions(original_image, prob, class_names)
# %%
'''INCEPTION_RASNET_V2'''
class InceptionResNetV2CardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(InceptionResNetV2CardClassifier, self).__init__()
        # Load Inception-ResNet-v2 pre-trained model
        self.base_model = timm.create_model('inception_resnet_v2', pretrained=True)
            
        # Replace the classifier layer
        self.base_model.classif = nn.Linear(self.base_model.classif.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)
# %%
inception_model = InceptionResNetV2CardClassifier()
# %%
inception_model.load_state_dict(torch.load(inception_model_path,weights_only=True, map_location=device))

#%%
inception_model.to(device)
# %%
original_image, image_tensor = preprocess_image2(test_image, transform2)
prob2 = predict(inception_model, image_tensor, device)
# %%
print(prob2)

#%%
# Find the index of the highest probability
predicted_index2 = prob2.argmax()  # or use torch.argmax(prob) if prob is a tensor

# Map the predicted index to the card label
predicted_card2 = class_labels[predicted_index2.item()]

print(f"The predicted card is: {predicted_card2}")

# %%
visualize_predictions(original_image, prob2, class_names)
# %%
'''IGNORE FOR NOW'''
'''NASNET_A'''

class NASNetACardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(NASNetACardClassifier, self).__init__()
        
        # Load NASNet-A model from timm with pretrained weights
        self.base_model = timm.create_model('nasnetalarge', pretrained=True)
        
        # Freeze all layers initially (optional, adjust as needed)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False
            
        # # Replace the classifier with a custom classifier
        nasnet_out_features = self.base_model.last_linear.in_features

        self.base_model.last_linear = nn.Sequential(
            nn.Linear(nasnet_out_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
#%%
nasnet_model = NASNetACardClassifier(num_classes=53)

nasnet_model.load_state_dict(torch.load(nasnet_model_path,weights_only=True, map_location=device))

nasnet_model.to(device)

#%%
prob3 = predict(nasnet_model, image_tensor, device)

predicted_index3 = prob3.argmax()  # or use torch.argmax(prob) if prob is a tensor

# Map the predicted index to the card label
predicted_card3 = class_labels[predicted_index3.item()]

print(f"The predicted card is: {predicted_card3}")

visualize_predictions(original_image, prob3, class_names)
# %%
