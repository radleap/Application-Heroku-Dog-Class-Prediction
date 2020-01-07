import os
from flask import Flask, flash, render_template, redirect, request, url_for, jsonify
from werkzeug.utils import secure_filename
import pickle
import torch.nn as nn
import torch
import io
import torchvision.transforms as transforms
from torchvision import models #getting pretrained models
from PIL import Image
from torch.autograd import Variable
import numpy as np
import cv2 #added for facial recognition

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),'images/') #where the images that are POST are stored
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

 ## Randomly selecting either the MobileV2 Model or SqueezeNet for added variation in results (why? more fun :))
# if np.random.random() >= 0.5:
#     #loading the SqueezeNet Model - main model for dog class prediction #uncomment if want
#     model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_Squeezenet_CNN_Transfer_20191212081002.pwf'))
#     model.eval()
# else:
#     # loading the MobileV2  Model - main model for dog class prediction
#     model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_MobileNet_CNN_Transfer_20191214165656.pwf'))
#     model.eval()

model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_Squeezenet_CNN_Transfer_20191212081002.pwf'))
model.eval()

# This model works fine on Heroku... testing the other
# model = torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_MobileNet_CNN_Transfer_20191214165656.pwf'))
# model.eval()

#loading the SqueezeNet Model - only detects if a dog or not returning True/False - loading pretrained from torchvision
model_detect_dog = models.squeezenet1_0(pretrained=True)
#model_detect_dog.load_state_dict(torch.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),'model_dog_detector__SqueezeNet_20200104162742.pwf')))
model_detect_dog.eval()

# Loading the dog classes, 133 total, from a static file
with open('static/classes.txt', 'rb') as file:
    classes = pickle.load(file)

# Setting up the flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "supertopsecretprivatekeyoobooboo"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Loading the initial "homepage" with upload.html template
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        # show the upload form
        return render_template('upload.html')

@app.route('/about', methods=['GET', 'POST'])
def about():
    if request.method == 'GET':
        return render_template('about.html')

#image loader transforms the image to a format for the neural network to use (as had been trained), loads image, apply, returns new image
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

# Human detection - uses opencv
def face_detector(file):
    face_cascade = cv2.CascadeClassifier('static/haarcascades/haarcascade_frontalface_alt.xml') #face detection xml model
    img = Image.open(file)
    img = np.array(img)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

# Dog detection: - applies image loaders, and the neural network to the image, returning True/False if dog detector
def dog_detector(img_path):
    image = image_loader(img_path)
    output = model_detect_dog(image) #use NN model_detect_dog
    _, preds_tensor = torch.max(output,1)
    preds = np.squeeze(preds_tensor.numpy())
    if preds >= 151 and preds <=268: #uses knowledge of the ImageNet classes
        x =True
    else:
        x = False
    return x

# Dog Class Prediction: applies image loaders, and the neural network to the image, returning the class name
def use_CNN(img_path):
    image = image_loader(img_path)
    output = model(image) #use NN
    _, preds_tensor = torch.max(output,1)
    preds = np.squeeze(preds_tensor.numpy())
    name = classes[preds.flat[0]] #labels contains the classes, in static files
    return(name)

# If the user uploads an image, this get the "POST" and applies the use_CNN (neural network to it)
# This code uses the static files (static due to Heroku documentation) and the predicted class to return the predicted class image
# The same POST is used by the squeezenet model to detect if a dog, and opencv to detect a human.
@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file'] #takes the name "file" from upload.html post form
        pred = use_CNN(file)
        dog_filename =  '\\static\\dog_class_images\\' + pred.capitalize().replace(" ", "_") +'.jpg'

        #dog detection squeezenet model application (true/false)
        is_dog = dog_detector(file)

        #face detection portion cv2 (true/false)
        is_human = face_detector(file) #checking for humans

        # Logic returning text based on detecting a dog,  detecting a human, or neither.
        if is_dog:
            return render_template('prediction.html', prediction_text = 'A dog was detected! And, the predicted dog class is a {}!'.format(pred), prediction_image = dog_filename)
        elif is_human:
            return render_template('prediction.html', prediction_text = 'A human was detected! But, this particular human looks more like a {}!'.format(pred), prediction_image = dog_filename)
        else:
            return render_template('prediction.html', prediction_text = 'Not sure what this creature is, but it looks like this dog breed {}. Suggest consulting the zoo.'.format(pred), prediction_image = dog_filename)

if __name__ == "__main__":
    app.run()
