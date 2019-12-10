import os
from flask import Flask, flash, render_template, redirect, request, url_for

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['SECRET_KEY'] = "supertopsecretprivatekey1234"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        # show the upload form
        return render_template('home.html')

    if request.method == 'POST':
        # check if a file was passed into the POST request
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']

        # if filename is empty, then assume no upload
        if image_file.filename == '':
            flash('No file was uploaded.')
            return redirect(request.url)

        # if the file is "legit"
        if image_file:
            passed = False
            try:
                filename = image_file.filename
                filepath = os.path.join(APP_ROOT,'images/')
                image_file.save(filepath)
                passed = True
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)

if __name__ == "__main__":
    app.run('127.0.0.1')
