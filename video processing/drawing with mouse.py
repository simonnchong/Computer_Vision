import cv2
import time

#* callback function rectangle
def draw_rectangle(event,x,y,flags,param):
    global pt1,pt2,top_left_clicked,bot_right_clicked
    
    if event == cv2.EVENT_LBUTTONDOWN:
        
        #* reset the rectangle
        if top_left_clicked == True and bot_right_clicked == True:
            pt1 = (0,0)
            pt2 = (0,0)
            top_left_clicked = False
            bot_right_clicked = False
        
        #* if click nothing yet
        if top_left_clicked == False:
            pt1 = (x,y)
            top_left_clicked = True
            
            
        #* if clicked top left but not bottom right
        elif bot_right_clicked == False:
            pt2 = (x,y)
            bot_right_clicked = True

#* global variable
pt1 = (0,0)
pt2 = (0,0)
top_left_clicked = False
bot_right_clicked = False

cap = cv2.VideoCapture(0) #* 0 means default camera

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#* connect to the callback
cv2.namedWindow("Test")
cv2.setMouseCallback("Test",draw_rectangle)

while True:
    
    ret,frame = cap.read()
    
    #* drawing on the frame based off the global variables
    if top_left_clicked: #* if this is true
        cv2.circle(frame,center=pt1,radius=5,color=(0,0,255),thickness=-1)
    
    if top_left_clicked and bot_right_clicked:
        cv2.rectangle(frame,pt1,pt2,(0,0,255),5)
    
    cv2.imshow("Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release() #* stop capturing
cv2.destroyAllWindows()