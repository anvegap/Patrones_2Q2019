import cv2
import numpy as np
import tensorflow as tf
import numpy as np
import warnings
import os

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2 as Net
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
#from tensorflow.python.util import deprecation

from keras.preprocessing import image
#from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential
from keras.models import Model

#from keras.optimizers import RMSprop

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

#from keras.callbacks import CSVLogger

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps 
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def createModel():
    
    #Clear TF graph to avoid clutter from old models or layers
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)

    #Define the size of input layer
    input_shape = (200, 200, 3)

    #Instanciate a model of type Sequential
    model = Sequential()

    #Add the input layer for 200x200x3 images
    #Add 2 convolutional layers with 32 3x3 kernel filters, using relu activation function
    #Enable padding for keeping same input size 
    #End by pooling the max value from 2x2 kernel filter
    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=input_shape, activation='relu'))
    model.add(Conv2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Add 2 convolutional layers with 64 3x3 kernel filters, using relu activation function
    #Enable padding for keeping same input size
    #End by pooling the max value from 2x2 kernel filter
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Add 2 convolutional layers with 128 3x3 kernel filters, using relu activation function
    #Enable padding for keeping same input size
    #End by pooling the max value from 2x2 kernel filter
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Add 2 convolutional layers with 256 3x3 kernel filters, using relu activation function
    #Enable padding for keeping same input size
    #End by pooling the max value from 2x2 kernel filter
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Conv2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #Add new layer that flattens (serialize) the input
    #Add a regular NN layer with output shape (256), using relu activation function
    #Add dropout layer to discard half of inputs at each update during training time
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    #Add a regular NN layer with output shape (256), using relu activation function
    #Add dropout layer to discard half of inputs at each update during training time
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    #Add a regular NN layer with output shape (1), using sigmoid activation function
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    #Load the pre-trained model weights
    model.load_weights('./model_weights.h5')

    return model
    
def show_camera(model):
    
    #Control variable for video stream
    pause = False
    
    #Create video stream
    cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        
        #Create window for showing images and results
        window_handle = cv2.namedWindow('CSI Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('CSI Camera', 640,460)
        
        while cv2.getWindowProperty('CSI Camera',0) >= 0:
            
            #Show video continuously
            if not pause:
                ret_val, img = cap.read()
                cv2.imshow('CSI Camera',img)
            
            #Capture any input from keyboard
            keyCode = cv2.waitKey(30) & 0xff
            
            #Stop the program on the ESC key
            if keyCode == 27:
               break
            
            #Resume video #b
            if keyCode == 98:
               pause = False

            #Stop video and grab last frame from stream #c
            if keyCode == 99:
               pause = True
               ret_val, img = cap.read()
               
               #Resize image according CNN's input layer shape
               resized_image = cv2.resize(img, (200, 200)) 
               
               #Convert image to numpy array before processing
               arr_img = image.img_to_array(resized_image)
               arr_img = np.expand_dims(arr_img, axis=0)
               arr_img = arr_img.astype('float32')
               arr_img/=255
               
               #Call CNN predictor
               prob = model.predict_proba(arr_img)
               

               #verify probabilities and assign output class
               if prob[0][0] > 0.5:
                   prob_label = str(round((prob[0][0]*100), 3)) + "% dog"
               else:
                   prob_label = str(round(((1-prob[0][0])*100), 3)) + "% cat"
                   
               #Overlap result and show input image    
               cv2.putText(img, prob_label, (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 2.6, (255,255,255), 1)
               cv2.imshow('CSI Camera',img)
                        
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        print ('Unable to open camera')

def show_image(img_path, model):
    
    #Load known image
    img = image.load_img(img_path)
    
    #Convert image to numpy array for processing
    arr_img = image.img_to_array(img)
    
    #Resize image according CNN's input layer shape
    arr_img = cv2.resize(arr_img, (200, 200))
    
    #Convert image to numpy array before processing
    arr_img = np.expand_dims(arr_img, axis=0)
    arr_img = arr_img.astype('float32')
    arr_img/=255

    #Call CNN predictor
    prob = model.predict_proba(arr_img)

    #Verify probabilities and assign output class
    if prob[0][0] > 0.5:
        prob_label = str(round((prob[0][0]*100), 3)) + "% dog"
    else:
        prob_label = str(round(((1-prob[0][0])*100), 3)) + "% cat"

    #Show result
    print(prob_label)
    
if __name__ == '__main__':
        
    #Build CNN architecture and load the pre-trained model weights
    mod = createModel()
    
    #Use CNN with camera input
    show_camera(mod)
    
    #Use CNN with known-local images
    #show_image('perro.jpeg', mod)
