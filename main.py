import cv2 as cv
import numpy as np
import time

def centerRectangle(x,y,w,h):
    centerX = x + int(w/2)
    centerY = y + int(h/2)
    return centerX, centerY

def detectCascade(frame, xmlModelMoto, xmlModelOto):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detectMoto = xmlModelMoto.detectMultiScale(gray,
                                           scaleFactor=1.1,
                                           minNeighbors=55,
                                           minSize=(30,45), 
                                           maxSize = (120, 180))
    detectOto = xmlModelOto.detectMultiScale(gray,
                                           scaleFactor=1.05,
                                           minNeighbors=30,
                                           minSize=(120,80), 
                                           maxSize = (360,240))
    
    
    vehicles = list(detectMoto)+list(detectOto)
    return vehicles

def DrawAndCount(frame, soLuongXe ,x,y,w,h):
    offset = 1.5
    p1 = (x, y)
    p2 = (x+w, y+h)
    centerPoint = centerRectangle(x, y, w, h)
    cv.circle(frame, centerPoint,2, (0,255,0),1)
    ptdt = 25/128*centerPoint[0] + 250
    if centerPoint[1] <= (ptdt + offset) and centerPoint[1] >= (ptdt - offset):
        soLuongXe += 1
    if w < h:
        cv.rectangle(frame, p1, p2, (0,0,255), 2)
        cv.putText(frame, "2 Banh", (x, y-20), cv.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)
    else:
        cv.rectangle(frame, p1, p2, (0,255,0), 2)
        cv.putText(frame, "4 Banh", (x, y-20), cv.FONT_HERSHEY_TRIPLEX,1,(255,244,0),2)
    return soLuongXe

def IoU(boxA, boxB):
	
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	
	iou = interArea / float(boxAArea + boxBArea - interArea)
	
	return iou

def main():
    soLuongXe = 0
    trackers = cv.legacy.MultiTracker_create()
    countFrame = 0 
    srcXmlModelMoto = "./models/CascadeMoto.xml"
    srcXmlModelOto = "./models/CascadeOto.xml"
    xmlModelMoto = cv.CascadeClassifier(srcXmlModelMoto)
    xmlModelOto = cv.CascadeClassifier(srcXmlModelOto)
    isTracking = 0
    
    video = cv.VideoCapture('./videos/BuoiSang.mp4')
    # video = cv.VideoCapture('./videos/BuoiToi.mp4')
    fps = 24
    prev = 0
    maxTrackingFrame = 24
    while True:
        timeElapsed = time.time() - prev
        if timeElapsed > 1./fps:
            prev = time.time()
            ret, frame =  video.read() 
            if not ret:
                print(' can not read video frame. Video ended?')
                break
            if isTracking == 0: # Bật Detect
                vehicles = detectCascade(frame, xmlModelMoto, xmlModelOto) # Lấy xe đã được detect
                ret, objs = trackers.update(frame) # Update lại tracker để lấy objs trước đó đã track được
                if vehicles != None:
                    trackers = cv.legacy.MultiTracker_create()
                    for x,y,w,h in vehicles:
                        isTracked = False
                        for obj in objs:
                            iou = IoU([x,y,x+w,y+h], [int(obj[0]),int(obj[1]),int(obj[0])+ int(obj[2]),int(obj[1])+int(obj[3])])
                            if iou > 0:
                                isTracked = True
                                break
                        if isTracked: # Nếu tọa độ của xe đã được track trước đó rồi, thì gán lại đúng obj đó vào lại bộ tracker
                            trackers.add(cv.legacy.TrackerCSRT_create(), frame, obj)
                        else: # Thêm obj vào tracker
                            roi = [x + int(w/8), y+int(h/12), w-int(w/4) ,h-int(h/6)]
                            trackers.add(cv.legacy.TrackerCSRT_create(), frame, roi)
                    isTracking = 1 # Bật tracking
            if isTracking == 1: # Bộ tracking được bật   
                if countFrame == maxTrackingFrame:
                    isTracking = 0
                    countFrame = 0
                ret, objs = trackers.update(frame)
                if ret:
                    for obj in objs:
                        soLuongXe = DrawAndCount(frame,soLuongXe, int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]))
                else:
                    print("tracking fail")
                    isTracking = 0
                countFrame = countFrame + 1
            
            cv.putText(frame, f"SO LUONG XE: {soLuongXe}", (450, 80), cv.FONT_HERSHEY_PLAIN, 2, (0,255,0),3)
            cv.line(frame,(0, 250), (1280, 500), (15, 233, 14), 3)
            cv.imshow("Video", frame)
            if cv.waitKey(1) == 113:
                break
        
    video.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    main()
    
