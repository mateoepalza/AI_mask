import cv2
import os
import numpy as np
import tensorflow as tf
import time

width = 160
height = 160

def main():
    classifier = define_classifier()
    model = load_model()
    video_capture(classifier, model)

def load_model():
    # Get the path of the neural network
    model_path = os.path.join(os.getcwd(), 'models', 'modelo2.hdf5')
    # load the model
    model = tf.keras.models.load_model(model_path)

    return model

def define_classifier():
    # path for the model
    prototxtPath = os.path.join(os.getcwd(), "models","deploy.prototxt")
    # path for the weights
    weightsPath = os.path.join(os.getcwd(), "models","res10_300x300_ssd_iter_140000.caffemodel")
    # 
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    return net

def detect(frame, classifier, model):
    # make a copy 
    frame_copy = np.copy(frame)
    (h, w) = frame.shape[:2]
    #change the color from RGB to grayscale
    #frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #frame_img = cv2.equalizeHist(frame_img)
    blob = cv2.dnn.blobFromImage(frame, 1.0 , (300, 300), (104.0, 177.0, 123.0))
    classifier.setInput(blob)
    faces = classifier.forward()

    # Iterate over all the faces
    for i in range(0, faces.shape[2]):
        #print("Enter")
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

            # get the face from the vector
            #face = frame_copy[y:(y+h),x:(x+w)]

            # prepare the face
            face = prep_face(face)
            # execute model
            res = execute_model(face, model) 
            # draw rectangle
            frame = draw_result(res, frame, (startX, startY, endX, endY))

    return frame 

def draw_result(result, frame, coord):

    font = cv2.FONT_HERSHEY_SIMPLEX

    if result[0][0] > result[0][1]:
        #red
        frame = cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0,0,255), 3)
        cv2.putText(frame,'Cliente sin tapabocas',(150,30),font,1,(0,0,255),2,cv2.LINE_AA)
    else:
        #green
        frame = cv2.rectangle(frame, (coord[0], coord[1]), (coord[2], coord[3]), (0,255,0), 3)
        cv2.putText(frame,'Cliente con tapabocas',(150,30),font,1,(0,255,0),2,cv2.LINE_AA)
    
    
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