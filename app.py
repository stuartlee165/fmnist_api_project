from flask import Flask, request
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

# Pytorch Code

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model_weights.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()


app = Flask(__name__)

auth = HTTPBasicAuth()

# In a real app, you would probably store your users in a database,
# but for simplicity, we'll just store them in a dictionary
users = {
    "user1": generate_password_hash("password1"),
    "user2": generate_password_hash("password2")
    # add more users as needed
}


# Flask API
@app.route("/upload", methods=["POST"])
def upload():
    # check if the post request has the file part
    if "image" not in request.files:
        return "No image part in the request", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    # if everything is okay, convert file to an image
    img = Image.open(file)

    transform = ToTensor()
    img_tensor = transform(img)

    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        pred = model(img_tensor)

    pred_string = classes[pred[0].argmax()]

    return f"Prediction: {pred_string}"


if __name__ == "__main__":
    app.run(debug=True)
