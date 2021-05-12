import face_recognition
import imutils
import pickle
import time
import cv2
import os
import dlib
dlib.DLIB_USE_CUDA = True
dlib.cuda.set_device(0)
 
#Find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

#Load the haarcascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)

#Load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read()) 
print("Streaming started")
video_capture = cv2.VideoCapture(0)

#Loop over frames from the video file stream
while True:

    #Grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    face_locations = face_recognition.face_locations(frame)

    #Convert the input frame from BGR to RGB 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #The facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb)
    names = []
    
    #Loop over the facial embeddings incase we have multiple embeddings for multiple fcaes
    for encoding in encodings:

        #Compare encodings with encodings in data["encodings"]
        #Matches contains array with boolean values and True for the embeddings it matches closely and False for rest
        matches = face_recognition.compare_faces(data["encodings"], encoding)

        #set name = unknown if no encoding matches
        name = "Unknown"

        # check to see if we have found a match
        if True in matches:

            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            #Loop over the matched indexes and maintain a count for each recognized face
            for i in matchedIdxs:

                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]

                #Increase count for the name we got
                counts[name] = counts.get(name, 0) + 1

            #Set name which has highest count
            name = max(counts, key=counts.get) 
 
        #Update the list of names
        names.append(name)

        #Loop over the recognized faces
        for ((y, x, h, w), name) in zip(face_locations, names):

            #Rescale the face coordinates
            #Draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, name, (w, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)

    #Display each frame
    cv2.imshow("Frame", frame)

    #Close each frame after 1 ms
    if cv2.waitKey(1) &amp; 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()