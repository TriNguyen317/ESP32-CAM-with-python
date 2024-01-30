import pandas as pd
import cv2
import urllib.request
import numpy as np
import os
from datetime import datetime
import face_recognition
from ultralytics import YOLO
import copy
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinter.simpledialog import askstring
import io

path = r'D:/Sinh_vien/Nam_4_HK1/AIOT/model/image_folder'
url='http://192.168.1.11/cam-mid.jpg'
model='D:/Sinh_vien/Nam_4_HK1/AIOT/model/best.pt'
camera = ['cam-lo.jpg', 'cam-hi.jpg', 'cam-mid']
##'''cam.bmp / cam-lo.jpg /cam-hi.jpg / cam.mjpeg '''
 



# Task detect image

def getboudingbox(model,img, conf=0.5, iou_thresh=0.45):
    result = model(img, iou=iou_thresh)

    boxes = result[0].boxes  # Boxes object for bbox outputs
    conf_detect = boxes.conf.cpu().numpy()
    box_detect = boxes.xyxy.cpu().numpy()
    idx = np.where(conf_detect > conf)
    return box_detect[idx], np.round_(conf_detect[idx], decimals=3)


# Draw
def detectImg(model,img):
    boxes, conf = getboudingbox(model,img)
    for num, i in enumerate(boxes):
        img = cv2.rectangle(img, (int(i[0]), int(i[1])), 
                        (int(i[2]), int(i[3])), (255, 0, 0), 2)
        cv2.putText(img, str(conf[num]), (int(i[0]), 
                        int(i[1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


 
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
 
def markAttendance(name):
    with open("Attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'/n{name},{dtString}')
 

def EmbeddingFace():
    images = []
    classNames = []
    myList = os.listdir(path)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    print(classNames)
    encodeListKnown = findEncodings(images)
    return encodeListKnown, classNames
#cap = cv2.VideoCapture(0)



class VideoApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # # Mở kết nối đến webcam
        # self.cap = urllib.request.urlopen(url)

        # Tạo canvas để hiển thị video
        self.canvas = tk.Canvas(window, width=840, height=640)
        self.canvas.pack()

        # Tạo nút để bắt đầu/stop stream
        self.btn_start = tk.Button(window, text="Start", command=self.start_stream)
        self.btn_start.pack(padx=20, pady=10, side=tk.LEFT)

        self.btn_stop = tk.Button(window, text="Stop", command=self.stop_stream)
        self.btn_stop.pack(pady=10, side=tk.LEFT)
        
        self.btn_save = tk.Button(window, text="Create new user", command=self.create_new_user)
        self.btn_save.pack(padx=20, pady=10, side=tk.LEFT)
        
        self.btn_delete = tk.Button(window, text="Delete User", command=self.delete_file)
        self.btn_delete.pack(pady=10, side=tk.LEFT)

        self.error_label = tk.Label(window, text="", fg="red", font=("Arial", 30) )
        self.error_label.pack(pady=10)
        
        self.streaming = False
        self.numUser = 30
        self.CurUser = len(os.listdir(path))
        self.embeddingface, self.classNames = EmbeddingFace()
        self.update()

    def start_stream(self):
        self.streaming = True
        self.btn_start["state"] = "disabled"
        self.btn_stop["state"] = "normal"
        self.error_label["text"] = ""
        

    def stop_stream(self):
        self.streaming = False
        self.btn_start["state"] = "normal"
        self.btn_stop["state"] = "disabled"
        #img = self.cap
    
    def capture_canvas(self):
        # Save canvas content as an image
        ps = self.canvas.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        return img
    
    def create_new_user(self):
        if self.CurUser < self.numUser:
        # Hiện hộp thoại để nhập tên người dùng
            self.stop_stream()
            user_name = askstring("Create New User", "Enter user name:")
            if user_name:

                img_resp=urllib.request.urlopen(url)
                imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
                img=cv2.imdecode(imgnp,-1)
                # img = captureScreen()
                # img = cv2.resize(img, (0, 0), None, 0.25, 0.25)                
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                boxes, _ = getboudingbox(model,img)
                print(boxes.shape)
                if boxes.shape == (0, 4):
                    print("Không tìm thấy khuôn mặt trong ảnh")
                    self.error_label["fg"]="red"
                    self.error_label["text"] = "Error: Không tìm thấy khuôn mặt trong ảnh"
                else:
                    cv2.imwrite('./image_folder/{}.jpg'.format(user_name), img)
                    print("New user created:", user_name)
                    self.error_label["fg"]="green"
                    self.error_label["text"] = "Thêm người dùng thành công"
                    self.CurUser += 1
                    self.embeddingface, self.classNames = EmbeddingFace()
            
    def delete_file(self):
    # Open a file dialog to select the file to delete
        self.stop_stream()
    
        file_path = filedialog.askopenfilename(title="Select a file to delete")

        if file_path:
            try:
                # Delete the selected file
                os.remove(file_path)
                self.CurUser -= 1
                print(f"File '{file_path}' deleted successfully.")
                self.error_label["fg"]="green"
                self.error_label["text"] = "Xóa thành công"
                self.embeddingface, self.classNames = EmbeddingFace()
                
            except Exception as e:
                print(f"Error: {e}")
                
    def update(self):
        if self.streaming:
            # Đọc frame
            img_resp=urllib.request.urlopen(url)
            imgnp=np.array(bytearray(img_resp.read()),dtype=np.uint8)
            img=cv2.imdecode(imgnp,-1)
            # img = captureScreen()
            #imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            boxes, _ = getboudingbox(model,img)
            #boxes = boxes/4
            boxes=boxes.astype(int)
            print("boxes: ", boxes)
            
            # facesCurFrame = face_recognition.face_locations(imgS)
            # print("facecurf: ", facesCurFrame)
            location = [(item[1], item[2], item[3], item[0]) for item in boxes]
            print("location: ", location)
            encodesCurFrame = face_recognition.face_encodings(imgS, location)
        
            for encodeFace, faceLoc in zip(encodesCurFrame, location):
                
                matches = face_recognition.compare_faces(self.embeddingface, encodeFace)
                faceDis = face_recognition.face_distance(self.embeddingface, encodeFace)
        # print(faceDis)
                matchIndex = np.argmin(faceDis)
                print("a: ", matchIndex)
                print("b: ", faceDis)
                print("c: ", matches)
                
                if matches[matchIndex]:
                    name = self.classNames[matchIndex].upper()
                    print("aaaa",name)
        # print(name)
                    y1, x2, y2, x1 = faceLoc
                    #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    markAttendance(name)
        
    

            # Chuyển đổi frame từ BGR sang RGB
            rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Tạo ảnh từ frame để hiển thị trên canvas
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)

            # Hiển thị ảnh trên canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.photo = photo

        # Gọi lại hàm update sau 10 milliseconds
        self.window.after(10, self.update)
    
    def __del__(self):
        # Giải phóng tài nguyên khi đối tượng bị hủy
        if hasattr(self, 'cap'):
            self.cap.release()

if 'Attendance.csv' in os.listdir(os.path.join(os.getcwd())):
    print("there iss..")
    os.remove("Attendance.csv")
    df=pd.DataFrame(list())
    df.to_csv("Attendance.csv")
else:
    df=pd.DataFrame(list())
    df.to_csv("Attendance.csv")
    


model = YOLO(model)
print('Encoding Complete')
# Tạo cửa sổ Tkinter
root = tk.Tk()

# Tạo đối tượng của ứng dụng video và chạy ứng dụng
app = VideoApp(root, "Video Streaming App")
root.mainloop()