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
video_capture = cv2.VideoCapture('TARGET_IMAGE/video.mp4')

frames = []
frame_count = 0

while video_capture.isOpened():
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Bail out when the video file ends
    if not ret:
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    frame = frame[:, :, ::-1]

    # Save each frame of the video to a list
    frame_count += 1
    frames.append(frame)

    # Every 128 frames (the default batch size), batch process the list of frames to find faces
    if len(frames) == 128:
        batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=0)

        # Now let's list all the faces we found in all 128 frames
        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            frame_number = frame_count - 128 + frame_number_in_batch
            print("I found {} face(s) in frame #{}.".format(number_of_faces_in_frame, frame_number))

            for face_location in face_locations:
                # Print the location of each face in this frame
                top, right, bottom, left = face_location
                print(" - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # Clear the frames array to start the next batch
        frames = []    #Display each frame
    #cv2.imshow("Frame", frame)

    #Close each frame after 1 ms
'''    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Release the video feed
for (name, value, *rest) in name_list:
    if (value):
        print(name)
video_capture.release()
cv2.destroyAllWindows()'''

