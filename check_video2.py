import face_recognition
import imutils
import pickle
import time
import cv2
import os
 
#Find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"

#Load the haarcascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)

#Load the known faces and embeddings saved in last file
data = pickle.loads(open('face_enc', "rb").read()) 
video_capture = cv2.VideoCapture('TARGET_IMAGE/video.mp4')
print("Streaming started")
name_list = {"unknown": False}
for name in data["names"]:
    name_list[name] = False
counter = 0
last_frame = []
last_name = ''

#Loop over frames from the video file stream
while video_capture.isOpened():
    counter+=1
    #Grab the frame from the threaded video stream
    ret, frame = video_capture.read()

    if(counter<1000 and (counter%25)==0):
        #Convert the input frame from BGR to RGB 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #The facial embeddings for face in input
        encodings = face_recognition.face_encodings(rgb)
        
        if(encodings!=[]):

            face_locations = face_recognition.face_locations(frame)
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
                name_list[name] = True
        
    #Display each frame
    #cv2.imshow("Frame", frame)

    #Close each frame after 1 ms
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the video feed
video_capture.release()
cv2.destroyAllWindows()

for (name, value) in name_list:
    if (value):
        print(name)