import os
import sqlite3

import cv2

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Cập nhật tên và ID vào CSDL
def insertOrUpdate(id, name):
    conn = sqlite3.connect("FaceRecognitionDB.db")
    cursor = conn.execute('SELECT * FROM People WHERE ID=' + str(id))
    isRecordExist = 0
    for row in cursor:
        isRecordExist = 1
        break

    if isRecordExist == 1:
        cmd = "UPDATE people SET Name=' " + str(name) + " ' WHERE ID=" + str(id)
    else:
        cmd = "INSERT INTO people(ID,Name) Values(" + str(id) + ",' " + str(name) + " ' )"

    conn.execute(cmd)
    conn.commit()
    conn.close()


id = input('Nhập MSSV: ')
name = input('Nhập tên SV: ')
print("[INFO] Bắt đầu chụp ảnh " + name + ", nhìn vào camera!")

insertOrUpdate(id, name)

sampleNum = 0

while (True):
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')
        if cv2.waitKey(1) & 0xFF == 99: #"C"
            sampleNum += 1
            cv2.imwrite("dataSet/User." + id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
    text_position = (img.shape[1] - 240, 30)
    cv2.putText(img, f"Images Captured: {sampleNum}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
    cv2.putText(img, "ESC: exit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "C: take photo", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Capturing Data ', img)
    if cv2.waitKey(50) & 0xFF == 27: #"ESC"
        print("\n[INFO] Đã thoát!")
        break
    elif sampleNum >= 20:
        print("\n[INFO] Chụp xong 20 ảnh của " + name + "!")
        break
cam.release()
cv2.destroyAllWindows()
