import cv2

cap = cv2.VideoCapture(0) #* 0 means default camera

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

write = cv2.VideoWriter("my_video.mp4",cv2.VideoWriter_fourcc(*"mpv4"),24,(width,height))

while True:
    
    ret,frame = cap.read()
    
    # gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    write.write(frame)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release() #* stop capturing
cv2.destroyAllWindows()