
# 1. Imports and Classname setup
import gradio as gr
import os
import torch

from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple , Dict

#Setup class names

class_names = ['Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Background_without_leaves',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy',
    'Corn___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn___Common_rust',
    'Corn___Northern_Leaf_Blight',
    'Corn___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy']

##2. Model and transforms preparations
effnetb2 , effnetb2_transforms = create_effnetb2_model(num_classes=39,)

#Load save weights
effnetb2.load_state_dict(
    torch.load(
        f="plantnet39.pth",
        map_location=torch.device("cpu"), #Load the model to the CPU
    )
)

### 3. Predict function ###
from typing import Tuple , Dict
from timeit import default_timer as timer

def predict(img) -> Tuple[Dict,float]: #Because we need Food class label and Prediction Time
     #Start timer
     start_time = timer()
     #Transform the input image for use with EffNetB2
     img = effnetb2_transforms(img).unsqueeze(0)
     #Put the model into eval mode , make prediction
     effnetb2.eval()
     with torch.inference_mode():
         # Pass transformed image through the model and turn the prediction logits into probabilites
         pred_probs = torch.softmax(effnetb2(img),dim=1)

     #Create a prediction label and prediction probability dictionary
     pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}

     #Calculate Pred Time
     end_time = timer()
     pred_time = round(end_time-start_time,4)
     #Return pred dict and pred time
     return pred_labels_and_probs , pred_time

###4. Gradio Interface ####

import gradio as gr

#Create title , description and article
title = "PlantNet39🌱"
description = "PlantNet39 is a deep learning model that classifies 39 different plant leaf diseases. Upload an image of a leaf, and the model will predict its disease class with high accuracy."
article = ""

#Create example list
# Create example list
example_list = [["examples/" + example] for example in os.listdir("examples")]
#Create the Gradio Demo
demo = gr.Interface(fn=predict,#maps inputs to outputs
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=39,label="Predictions"),
                             gr.Number(label="Prediction time (s)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

#Launch the demo
demo.launch(debug=False, #Print errors locally
            share=True) #Generate a Publically shareable URL
