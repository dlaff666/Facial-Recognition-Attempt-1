import face_recognition
import imutils
import pickle
import time
import cv2
import os
import math

#Find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

#Load the haarcascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)

#Load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read()) 
data_encodings = data["encodings"]
data_names = data["names"]
video_capture = cv2.VideoCapture(0)

counter = 0
check_frequency = 10
last_frame = []
last_name = ''
print("Streaming started")

#Loop over frames from the video file stream
while video_capture.isOpened():
    counter+=1
    #Grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    frame2 = cv2.resize(frame, (0,0), fx=0.29, fy=0.29)
    frame2 = frame2[:, :, ::-1]
    names = []

    if(counter%check_frequency == 0):
        #Convert the input frame from BGR to RGB 
        #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #The facial embeddings for face in input
        face_locations = face_recognition.face_locations(frame2)
        last_frame = []
        
        if(face_locations!=[]):
            encodings = face_recognition.face_encodings(frame2,face_locations)
            #Loop over the facial embeddings incase we have multiple embeddings for multiple fcaes
            for encoding in encodings:

                #Compare encodings with encodings in data["encodings"]
                #Matches contains array with boolean values and True for the embeddings it matches closely and False for rest
                matches = face_recognition.compare_faces(data_encodings, encoding)

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
                        name = data_names[i]

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
                    x*=(100/29)
                    y*=(100/29)
                    h*=(100/29)
                    w*=(100/29)
                    x=math.floor(x)
                    y=math.floor(y)
                    h=math.floor(h)
                    w=math.floor(w)
                    cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    this_frame = [y,x,h,w,name]
                    last_frame.append(this_frame)
                        
    else:
        for (y, x, h, w, name) in last_frame:

            #Rescale the face coordinates
            #Draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            cv2.putText(frame, name, (w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    if(not counter%check_frequency == 0):
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    #Display each frame
    cv2.imshow("Frame", frame)

    #Close each frame after 1 ms

#Release the video feed
video_capture.release()
cv2.destroyAllWindows()

