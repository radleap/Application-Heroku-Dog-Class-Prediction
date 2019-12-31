import os
from flask import Flask, flash, render_template, redirect, request, url_for, jsonify
from werkzeug.utils import secure_filename
import pickle
import torch.nn as nn
import torch
import io
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import numpy as np

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'images/')
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

#loading the machine learning model
model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_Squeezenet_CNN_Transfer_20191214165952.pwf'))
model.eval()

with open('static\\classes.txt', 'rb') as file:
    classes = pickle.load(file)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "supertopsecretprivatekeyoobooboo"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # show the upload form
        return render_template('upload.html')


def image_loader(image_name):
    transformer = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    image = Image.open(image_name)
    image = transformer(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

def use_CNN(img_path):
    image = image_loader(img_path)
    output = model(image) #use NN
    _, preds_tensor = torch.max(output,1)
    preds = np.squeeze(preds_tensor.numpy())
    name = classes[preds.flat[0]] #labels contains the classes
    return(name)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file'] #takes the name "file" from upload.html post form
        pred = use_CNN(file)
        dog_filename =  '\\static\\dog_class_images\\' + pred.capitalize().replace(" ", "_") +'.jpg'
        #return jsonify({'prediction': pred})
        #return render_template('prediction.html', prediction_text = 'The dog class is {}'.format(pred), prediction_image = dog_filename)
        return render_template('prediction.html', prediction_text = 'The dog class is {}'.format(pred), prediction_image = dog_filename)

# if __name__ == "__main__": #needed to run locally
#    app.run(port = 4555, debug = True) # hide this to run on Heroku?
if __name__ == "__main__":
    app.run()
