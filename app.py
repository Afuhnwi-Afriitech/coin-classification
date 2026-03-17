from flask import Flask, render_template, request
import os

from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET","POST"])
def index():

    coin, confidence = None, None

    if request.method == "POST":

        file = request.files["file"]
        path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(path)

        coin, confidence = predict_image(path)

    return render_template("index.html", coin=coin, confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)