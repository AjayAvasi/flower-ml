from train import device, load_model
import torch
import flask
import torchvision
import vector_db
import os
import csv

cnn, train_losses, train_accuracy, valid_losses, valid_accuracy = load_model("./models/2024-05-05_20-13-40")
cnn = cnn.to(device)
cnn.eval()

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToPILImage(),
    torchvision.transforms.Resize((180, 180)),
    torchvision.transforms.ToTensor()
])

app = flask.Flask(__name__)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
classes = ["astilbe", "bellflower", "black_eyed_susan", "calendula", "california_poppy", "carnation",
           "common_daisy", "coreopsis", "daffodil", "dandelion", "iris", "magnolia", "rose",
           "sunflower", "tulip", "water_lily"]


@app.route("/search-and-predict", methods=["POST"])
def search_and_predict():
    file = flask.request.files["img"]
    if file.filename.split(".")[-1].lower() not in ALLOWED_EXTENSIONS:
        return flask.jsonify({"error": "Invalid file type"})
    file.save("./temp/temp.jpg")
    max_prediction, predictions = picture_to_prediction("./temp/temp.jpg")
    k = 1
    if "k" in flask.request.args:
        k = int(flask.request.args["k"])
    closest = vector_db.get_k_closest(k, predictions)
    return flask.jsonify({"class_index": max_prediction, "vector": predictions, "flower": classes[max_prediction],
                          "similar": closest.to_dict(orient="records")})


@app.route("/flowers/<path:image_path>", methods=["GET"])
def get_image(image_path):
    image_path = os.path.join("./flowers", image_path)
    if os.path.exists(image_path):
        return flask.send_file(image_path)
    return "{} does not exist".format(image_path)

@app.route("/", methods=["GET"])
def index():
    return flask.render_template("index.html")


def picture_to_prediction(path):
    img = torchvision.io.read_image(path)
    img = transform(img)
    img = img[0:3, :, :]
    img = img.view(1, 3, 180, 180).to(device)
    with torch.no_grad():
        predictions = cnn(img.to(device))
    predictions = predictions[0]
    max_prediction = torch.argmax(predictions)
    return max_prediction.item(), predictions.tolist()


def add_all_vectors():
    data = []
    with open("data.csv", "r") as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            img_path = row[0]
            if img_path == "path":
                continue
            vector = picture_to_prediction(img_path)[1]
            data.append({"vector": vector, "img_path": img_path})
            print(f"Added {i}")
            i += 1
    vector_db.add_vectors(data)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
