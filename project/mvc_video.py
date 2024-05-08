
import cv2
import numpy as np
import imutils
from imutils import perspective
from imutils import contours

# 初始化视频捕捉
#cap = cv2.VideoCapture(".\\images\\gap\\redtestmp44.mp4")
cap = cv2.VideoCapture(".\\video\\Video_3.avi") 

def nothing(x):
    pass

if not cap.isOpened():
    print("无法打开摄像头")
    exit()
    
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建一个VideoWriter对象,指定输出文件名、编码格式、帧率和分辨率
out = cv2.VideoWriter(".\\video\\output_3.avi", cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))


cv2.namedWindow('setting')
cv2.resizeWindow('setting', width=800, height=600)
cv2.moveWindow('setting', 50, 50)
cv2.createTrackbar('detectgap','setting',300,2000,nothing)
cv2.createTrackbar('detectx1','setting',1350,frame_width,nothing)
cv2.createTrackbar('detecty1','setting',2460,frame_height,nothing)
cv2.createTrackbar('detectx2','setting',1650,frame_width,nothing)
cv2.createTrackbar('detecty2','setting',2600,frame_height,nothing)
cv2.createTrackbar('cannythreshold1','setting',15,300,nothing)
cv2.createTrackbar('cannythreshold2','setting',40,600,nothing)

imageindex = 0

while(True):
    ret, frame = cap.read()
    if frame is None:
        break
    
    detectgap=cv2.getTrackbarPos('detectgap','setting') 
        
    detectx1=cv2.getTrackbarPos('detectx1','setting')
    detecty1=cv2.getTrackbarPos('detecty1','setting')
    detectx2=cv2.getTrackbarPos('detectx2','setting')
    detecty2=cv2.getTrackbarPos('detecty2','setting')    

    cannythreshold1 = cv2.getTrackbarPos('cannythreshold1','setting')
    cannythreshold2 = cv2.getTrackbarPos('cannythreshold2','setting')

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #lower_red1 = np.array([0, 120, 70])
    #upper_red1 = np.array([10, 255, 255])
    #lower_red2 = np.array([170, 120, 70])
    #upper_red2 = np.array([180, 255, 255])
    
    # 根据阈值获取红色区域的掩膜
    #mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    #mask = cv2.bitwise_or(mask1, mask2)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    mask = np.zeros_like(gray)
    cv2.fillPoly(mask, np.array([[[detectx1 ,detecty1], [detectx2, detecty1],  [detectx2, detecty2],[detectx1, detecty2]]]), color=255)
    mask = cv2.bitwise_and(gray, mask)    
                
    # 执行灰度变换
    #gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # 执行高斯滤波
    gray = cv2.GaussianBlur(mask, (7, 7), 0)

    # 执行Canny边缘检测
    edged = cv2.Canny(gray, cannythreshold1, cannythreshold2)
    # 执行腐蚀和膨胀后处理
    edged = cv2.dilate(edged, None, iterations=2)
    edged = cv2.erode(edged, None, iterations=2)

    # 在边缘映射中寻找轮廓
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # 对轮廓点进行排序
    (cnts, _) = contours.sort_contours(cnts, method="bottom-to-top")
    
    cv2.rectangle(frame, (detectx1, detecty1), (detectx2, detecty2), (128, 128, 128), 2)
    
    # 遍历轮廓点
    index = 0
    for (i, c) in enumerate(cnts):
        # 计算轮廓点矩形框
        (x, y, w, h) = cv2.boundingRect(c)
        # 绘制矩形框
        if h > 30:
            print(x, y, w, h)
            print(index)
            if index == 0 and detectx1 < x and detectx2 > x + w and detecty2 > y:
                x1 = x
                y1 = y
                w1 = w
                h1 = h
                index = index + 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 0), 2)                
            elif index == 1 and detectx1 < x and detectx2 > x + w and detecty1 < y + h:
                dist = abs(y + h - y1)
                print(dist,end=' ')
                print('pix')
                print(dist/52,end=' ')
                print(end='mm')
                distmm = dist/52
                if distmm < detectgap/1000:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)
                    
                cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 0), 2)                  
             
                cv2.rectangle(frame, (x1 - 100, y1), (x + w + 100, y + h), color, 2)
                diststr = str(round(dist/76.933, 2)) + 'mm'
                cv2.putText(frame, diststr, (x + w, y), cv2.FONT_HERSHEY_COMPLEX, 3, color, 2, cv2.LINE_AA)
                index = index + 1
                break
            
            
        
    resized_image = cv2.resize(frame, None,  fx=0.2, fy=0.2,interpolation = cv2.INTER_AREA)
    resized_gray = cv2.resize(gray, None,  fx=0.2, fy=0.2,interpolation = cv2.INTER_AREA)
    resized_edged = cv2.resize(edged, None,  fx=0.2, fy=0.2,interpolation = cv2.INTER_AREA)
    # 显示图像
    cv2.imshow('Image', resized_image)
    cv2.imshow('gray', resized_gray)
    cv2.imshow('edged', resized_edged)
    cv2.imwrite(".//video//video3//" + str(imageindex) +'.jpg', frame)
    out.write(frame)
    imageindex += 1

    keyboard = cv2.waitKey(400)
    if keyboard == 'q' or keyboard == 27:
        print('Quit')
        break

# 释放捕捉器
cap.release()
cv2.destroyAllWindows()