#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 17:08:14 2022

@author: haiduozhao
"""

import PySimpleGUIWeb as sg
import cv2
import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub 
import numpy as np
from PIL import Image
import tensorflow as tf

filename=""

left_layout = [[sg.Text("Welcome to Flower Recognition System.")], 
          [sg.Text("You can choose to upload a picuter:")],
          [sg.InputText("Please enter image path", key = "dir_name"), sg.Button("Find", enable_events=True)],
          [sg.Text("Or take a photo:"), sg.Button("Take a photo", enable_events=True)],
          [sg.Image(size=(224,224), key = "IMAGE")],
          [sg.Button("Submit", enable_events=True), sg.Button("Clear", enable_events=True)]]

right_layout = [[sg.Text("Recognise results:")],
                [sg.Text(size=(20,2), key="NAME")], 
                [sg.Text(size=(20,2), key="ACC", text_color="lightgreen")]]

layout = [[sg.Column(left_layout),
          sg.Column(right_layout, element_justification="center")]]

# Create the window
window = sg.Window("Flower Recognition", layout, element_justification="center")   

def clear():
    window["Take a photo"].update(disabled=False)
    window["Find"].update(disabled=False)
    window["dir_name"].update("")
    window["IMAGE"].update(size=(224,224), filename="background.png")
    window["NAME"].update(" ")
    window["ACC"].update(" ")

while True:  # Event Loop
    event, values = window.read()

     # Create events
    if event == "Find":
        window["IMAGE"].update(size=(224,224), filename="background.png")
        filename = values["dir_name"]
        if filename == "":
            sg.popup_ok("Please enter an image path")
        elif filename.endswith(".jpg")==False | filename.endswith(".png")==False | filename.endswith(".jpeg") == False:
            window["Find"].update(disabled=True)
            window["Take a photo"].update(disabled=True)
            window["Submit"].update(disabled=True)
            window["Clear"].update(disabled=True)
            sg.popup_ok("Please choose a picture in correct format.")
            clear()
            window["Submit"].update(disabled=False)
            window["Clear"].update(disabled=False)
            window["dir_name"].update("")
            window["IMAGE"].update("background.png", size=(224,224))
            filename = ""
        else:
            pic = cv2.imread(filename)
            img = cv2.resize(pic, (224,224), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite("img.png", img)
            window["NAME"].update("")
            window["ACC"].update("")
            window["IMAGE"].update("img.png")
            window["Take a photo"].update(disabled=True)
    
    if event == "Clear":
        clear()
        filename = ""
        
    if event == "Submit":
        #search info from the internet and display the content
        window["ACC"].update("")
        
        try:
            if filename=="":
                sg.popup_ok("There is no picture.")
            else:
                model = keras.models.load_model("my_model_mobilenet.h5", custom_objects={'KerasLayer':hub.KerasLayer})
                im = Image.open(filename)
                test_image = np.asarray(im)
                image = np.squeeze(test_image)
                image = tf.image.resize(image, (224, 224))/255.0
                prediction = model.predict(np.expand_dims(image, axis=0))
                top_values, top_indices = tf.math.top_k(prediction, 5)
                acc = top_values.numpy()[0][0]
                label = top_indices[0][0]
                with open("labels.txt") as f:
                    name = f.readlines()[label]
                if acc < 0.5:
                    window["NAME"].update("Cannot recognise the picture.", text_color="red")
                else:
                    window["NAME"].update("Predic flower: "+name)
                    window["ACC"].update(str("Predic accuracy: "+'%.1f'%(acc*100) + " %"))  
        except FileNotFoundError:
             sg.popup_ok("The image path is invalid.")
                             
    if event == "Take a photo":
        #use camare to take a photo 
        cv2.namedWindow('camera')
        cap = cv2.VideoCapture(0)
        ret,frame = cap.read()
        img = cv2.resize(frame, (224,224), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite("img.png", img)
        filename = os.path.abspath("img.png")
        cap.release()
        cv2.destroyWindow("camera")
        window["IMAGE"].update("img.png")
        window["Find"].update(disabled=True)
        window["Take a photo"].update(disabled=True)

    # End program if user closes window
    if event == None and 'Exit':
        try:
            os.remove("img.png")
        except FileNotFoundError:
            break
        break

window.close()
