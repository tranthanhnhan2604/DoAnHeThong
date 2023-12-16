import cv2
import numpy as np
from PIL import Image
import pickle
import sqlite3
import tensorflow as tf
import numpy as np

# Khởi tạo bộ phát hiện khuôn mặt
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
save_model = tf.keras.models.load_model("recognizer/trainer.h5")

id = 0
#set text style
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (0,255,0)
fontcolor1 = (0,0,255)

# Hàm lấy thông tin người dùng qua ID
def getProfile(id):
    conn = sqlite3.connect("FaceRecognitionDB.db")
    cursor = conn.execute("SELECT * FROM People WHERE ID=" + str(id))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

cam = cv2.VideoCapture(0);

while(True):
    ret,img = cam.read();
    # Lật ảnh
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceDetect.detectMultiScale(gray, 1.1, 5);
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        roi_gray = gray[y:y+h, x:x+w]       #Cắt phần ảnh xám tương ứng với khuôn mặt.
        roi_gray = cv2.resize(src=roi_gray, dsize=(100,100))        #Resize ảnh xám thành kích thước 100x100
        roi_gray = roi_gray.reshape((100,100,1))        #Reshape ảnh xám thành kích thước (100, 100, 1)
        roi_gray = np.array(roi_gray)       #Chuyển đổi ảnh xám thành một mảng numpy.
        result = save_model.predict(np.array([roi_gray]))   #Sử dụng mô hình đã được lưu trữ để dự đoán lớp của ảnh.
        finalLabel = np.argmax(result)      #Lấy chỉ số của lớp có xác suất cao nhất từ kết quả dự đoán.

        profile = None
        
        if finalLabel == 0:
            profile = getProfile(1)
            cv2.putText(img, "MSSV: " + str(profile[0]), (x,y+h+30), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Ten: "  + str(profile[1]), (x,y+h+60), fontface, fontscale, fontcolor ,2)
        if finalLabel == 1:
            profile = getProfile(2)
            cv2.putText(img, "MSSV: " + str(profile[0]), (x,y+h+30), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Ten: "  + str(profile[1]), (x,y+h+60), fontface, fontscale, fontcolor ,2)
        if finalLabel == 2:
            profile = getProfile(58)
            cv2.putText(img, "MSSV: " + str(profile[0]), (x,y+h+30), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Ten: "  + str(profile[1]), (x,y+h+60), fontface, fontscale, fontcolor ,2)
        if finalLabel == 3:
            profile = getProfile(73)
            cv2.putText(img, "MSSV: " + str(profile[0]), (x,y+h+30), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Ten: "  + str(profile[1]), (x,y+h+60), fontface, fontscale, fontcolor ,2)
        if finalLabel == 4:
            profile = getProfile(74)
            cv2.putText(img, "MSSV: " + str(profile[0]), (x,y+h+30), fontface, fontscale, fontcolor ,2)
            cv2.putText(img, "Ten: "  + str(profile[1]), (x,y+h+60), fontface, fontscale, fontcolor ,2)


    text_position = (10,30)
    cv2.putText(img, "ESC to exit", text_position, fontface, 0.8, fontcolor1, 2)
    cv2.imshow('Nhan dien khuon mat',img)
    # Nếu nhấn ESC thì thoát
    if cv2.waitKey(1) == 27:
        break;
cam.release()
cv2.destroyAllWindows()

