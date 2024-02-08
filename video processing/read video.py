import cv2
import time

cap = cv2.VideoCapture("../DATA/hand_move.mp4")

if cap.isOpened() == False:
    print("File not found!")

while cap.isOpened():
    
    ret,frame = cap.read()
    
    if ret == True:
        
        # time.sleep(1/20) #control the speed
        cv2.imshow("Frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    else:
        break

cap.release() #* stop capturing
cv2.destroyAllWindows()