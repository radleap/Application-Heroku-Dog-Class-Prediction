import os
from flask import Flask, flash, render_template, redirect, request, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'images/')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "supertopsecretprivatekeyoobooboo"

#Loading Model
model = pickle.load(open('model-model-scratch-20191208-0848.pkl','rb'))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # show the upload form
        return render_template('upload.html')

@app.route("/upload", methods = ['GET','POST'])
# def upload():
#     target = os.path.join(APP_ROOT,'images/')
#
#     if not os.path.isdir(target):
#         os.mkdir(target)
#
#     uploaded_file = request.files("file")
#     destination = "/".join([target,'123'])
#     uploaded_file.save(destination)
#     # for file in request.files.getlist("file"):
    #     filename = file.filename
    #     destination = "/".join([target,filename])
    #     file.save(destination)
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('upload', filename=filename))
    # return '''
    # <!doctype html>
    # <title>Upload new File</title>
    # <h1>Upload new File</h1>
    # <form method=post enctype=multipart/form-data>
    #   <input type=file name=file>
    #   <input type=submit value=Upload>
    # </form>
    # '''
    return render_template("complete.html")

if __name__ == "__main__":
    app.run(port = 4555, debug = True)
