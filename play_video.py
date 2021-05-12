import cv2
video_capture = cv2.VideoCapture('TARGET_IMAGE/video.mp4')

while True:
    ret, frame = video_capture.read()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    cv2.imshow("Frame", frame)
