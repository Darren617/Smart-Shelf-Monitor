from ultralytics import YOLO
import cv2
import numpy as np
model=YOLO('yolov8n.pt')

capture =cv2.VideoCapture(0,cv2.CAP_DSHOW)

capture.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

if not capture.isOpened():
    raise RuntimeError("摄像头打开失败")

print("按q退出程序！")

while True:
    ret,frame=capture.read()
    if not ret:
        break

   # result=model(frame,verbose=False)[0]
    result=model.track(frame, persist=True)[0]


    
    bottle_centers=[]
    names=model.names
    boxes=result.boxes
    if boxes is not None and len(boxes)>0:
        xyxy=boxes.xyxy.cpu().numpy().astype(int)
        clss=boxes.cls.cpu().numpy().astype(int)
        confs=boxes.conf.detach().cpu().numpy()
        
        
        ids=boxes.id
        ids_np=None
        if ids is not None:
            ids_np=ids.detach().cpu().numpy().astype(int)

        cup_ids=0
        for idx,(box,cls_id,conf) in enumerate(zip(xyxy,clss,confs)):
            
            x1,y1,x2,y2=map(int,box)
            cx=int((x1+x2)/2)
            cy=int((y1+y2)/2)
            label=names.get(int(cls_id),str(cls_id))

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.circle(frame,(cx,cy),4,(0,0,255),-1)

            if ids_np is None:
                bid=None
            else:
                bid=int(ids_np[idx])


            if bid is not None:
                id_str=f"ID={bid}" 
            else:
                "ID=None" 
            
            if label=="bottle":
                
                bottle_centers.append((cx,cy))
                position=f"{label} {id_str} {conf: .2f} | center_position({cx},{cy})"
            else:
                position=f"{label} {id_str} {conf: .2f} | center_position({cx},{cy})"

            cv2.putText(frame,position,(x1,max(20,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,225,0),2)
            print(position)
        if bottle_centers:
            cv2.putText(frame,f"BOTTLE: {len(bottle_centers)}",(12,29),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3)

    cv2.imshow("pres 'q' to quit",frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break


capture.release()
cv2.destroyAllWindows()