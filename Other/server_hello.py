from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Hello, internet!"

if __name__ == "__main__":
    app.run('127.0.0.1')