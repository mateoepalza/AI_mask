import cv2
import os
import numpy as np
import tensorflow as tf
import time

# width and height of the image (pixels)
width = 160
height = 160

def main():
    # load the classifier
    classifier = define_classifier()
    # load the trained model
    model = load_model()
    # creates a mapping pixel for the gamma values
    mapping = creates_mapping(2)
    # capture the video 
    video_capture(classifier, model, mapping)

def load_model():
    # Get the path of the neural network
    model_path = os.path.join(os.getcwd(), 'models', 'modelo5.hdf5')
    # load the model
    model = tf.keras.models.load_model(model_path)

    return model

def define_classifier():
    # path for the model
    prototxtPath = os.path.join(os.getcwd(), "models","deploy.prototxt")
    # path for the weights
    weightsPath = os.path.join(os.getcwd(), "models","res10_300x300_ssd_iter_140000.caffemodel")
    # create the net with the structure and the weights
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    return net

def creates_mapping(gamma):
    # build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = (1.0 / gamma)
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table
    

def detect(frame, classifier, model, mapping):
    # make a copy 
    frame_copy = np.copy(frame)
    # obtain the width and height of the image
    (h, w) = frame.shape[:2]
    # change the format of the image from 
    blob = cv2.dnn.blobFromImage(frame, 1.0 , (300, 300), (104.0, 177.0, 123.0))
    # Input the image into the classifier
    classifier.setInput(blob)
    # retrieve all the faces found in the image
    faces = classifier.forward()

    # Iterate over all the faces
    for i in range(0, faces.shape[2]):
        # check the confidence in each face
        confidence = faces[0,0,i,2]

        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame_copy[startY:endY, startX:endX]

            # prepare the face
            face = prep_face(face, mapping)
            # execute model
            res = execute_model(face, model) 
            # draw rectangle
            frame = draw_result(res, frame, (startX, startY, endX, endY))

    return frame 

def draw_result(result, frame, coord):

    # Define a font for the text
    font = cv2.FONT_HERSHEY_PLAIN

    if result[0][0] > result[0][1]:
        #red
        frame = cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0,0,255), 2)
        cv2.putText(frame,'Sin tapabocas',(coord[0],coord[1]-6),font,(2*((coord[3]-coord[0])+(coord[2]-coord[1])))/500,(0,0,255),2,cv2.LINE_AA)
    else:
        #green
        frame = cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0,255,0), 2)
        cv2.putText(frame,'Con tapabocas',(coord[0],coord[1]-6),font,(2*((coord[3]-coord[0])+(coord[2]-coord[1])))/500,(0,255,0),2,cv2.LINE_AA)
    
    
    return frame

def execute_model(face, model):
    # Predict based on an entry
    result = model.predict(face)

    return result

def prep_face(face, mapping):
    # modify the width and height of the image
    image_f = cv2.resize(face, (width, height))
    # Adjust the gamma of the image
    image_f = cv2.LUT(image_f, mapping)
    # change the color from RGB to grayscale
    image_f = cv2.cvtColor(image_f, cv2.COLOR_BGR2GRAY)
    image_f = cv2.equalizeHist(image_f)
    # reshape
    image_f = np.reshape(image_f, (-1, width, height, 1))
    # normalize the image
    image_f = image_f/255

    return image_f

def video_capture(classifier, model, mapping):
    # the argument is the "position" of the camera
    cam = cv2.VideoCapture(0)

    while(True):
        # Capture frame by frame
        ret, frame =  cam.read()

        # In here we should do our operations
        frame = detect(frame, classifier, model, mapping)

        # Display the resulting frame
        cv2.imshow('frame',frame)

        time.sleep(0.025)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()