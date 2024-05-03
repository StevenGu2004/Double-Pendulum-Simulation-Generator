from flask import Flask, render_template, request
from waitress import serve

app = Flask(__name__)


@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")


if __name__ == "__main__":
    # NOTE: Delete the previous terminal when running the server

    # print("Updating server...") for some reason printing breaks everything :/
    serve(app, host="0.0.0.0", port=8000)
    # print("Finished serving server.")
