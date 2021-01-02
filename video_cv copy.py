import cv2
import os
import numpy as np
import tensorflow as tf
import time

width = 80
height = 80

def main():
    classifier = define_classifier()
    model = load_model()
    video_capture(classifier, model)

def load_model():
    # Get the path of the neural network
    model_path = os.path.join(os.getcwd(), 'models', 'modelo.hdf5')
    # load the model
    model = tf.keras.models.load_model(model_path)

    return model

def define_classifier():
    # get the path of the classifier
    face_classifier = os.path.join(os.getcwd(),'models','frontal_face.xml')
    # Activate the face classifier
    face = cv2.CascadeClassifier(face_classifier)

    return face

def detect(frame, classifier, model):
    # make a copy 
    frame_copy = np.copy(frame)
    #change the color from RGB to grayscale
    frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    frame_img = cv2.equalizeHist(frame_img)

    # Detect faces
    faces = classifier.detectMultiScale(frame_img)
    # Iterate over all the faces
    for (x,y,w,h) in faces:
        #print("Enter")
        # get the face from the vector
        face = frame_copy[y:(y+h),x:(x+w)]
        # prepare the face
        face = prep_face(face)
        # execute model
        res = execute_model(face, model) 
        # draw rectangle
        frame = draw_result(res, frame, (x,y,w,h))

    return frame 

def draw_result(result, frame, coord):

    if result[0][0] > result[0][1]:
        #red
        frame = cv2.rectangle(frame, (coord[0], (coord[1]+coord[3])), ((coord[0]+coord[2]), coord[1]), (255,0,0))
    else:
        #green
        frame = cv2.rectangle(frame, (coord[0], (coord[1]+coord[3])), ((coord[0]+coord[2]), coord[1]), (0,255,0))
    
    
    return frame

def execute_model(face, model):
    # Predict based on an entry
    result = model.predict(face)

    return result

def prep_face(face):
    # modify the width and height of the image
    image_f = cv2.resize(face, (width, height))
    # change the color from RGB to grayscale
    image_f = cv2.cvtColor(image_f, cv2.COLOR_BGR2GRAY)
    # reshape
    image_f = np.reshape(image_f, (-1, width, height, 1))
    # normalize the image
    image_f = image_f/255

    return image_f

def video_capture(classifier, model):
    # the argument is the "position" of the camera
    cam = cv2.VideoCapture(0)

    while(True):
        # Capture frame by frame
        ret, frame =  cam.read()

        # In here we should do our operations
        frame = detect(frame, classifier, model)

        # Display the resulting frame
        cv2.imshow('frame',frame)

        #time.sleep()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything done, release the capture
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()