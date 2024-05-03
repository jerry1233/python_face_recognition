import face_recognition
import cv2

# 加载特定人脸图像
known_image = face_recognition.load_image_file("face_image5.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# 开启摄像头
video_capture = cv2.VideoCapture(0)

while True:
    try:
        # 捕获每一帧图像
        ret, frame = video_capture.read()

        # 将每一帧图像转换为RGB格式（face_recognition库要求）
        #rgb_frame = frame[:, :, ::-1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 在当前帧中查找所有的人脸及其编码
        face_locations = face_recognition.face_locations(rgb_frame) #在摄像头图像中定位人脸的位置
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations) #提取摄像头图像中检测到的人脸的特征编码

        # 遍历每个检测到的人脸
        for face_encoding in face_encodings:
        
            # 对比当前人脸与特定人脸的编码
            matches = face_recognition.compare_faces([known_encoding], face_encoding)
            if True in matches:
                print("发现特定人脸！")
            
            #计算人脸编码之间的欧几里得距离
            #face_recognition.face_distance() 计算人脸编码之间的欧几里得距离。它衡量两张脸有多相似，较低的距离表示更高的相似度。
            #在这里，它计算了已知人脸编码 (known_encoding) 和当前人脸编码 (face_encoding) 之间的距离，后者是从摄像头视频流中检测到的人脸得到的。
            face_distance = face_recognition.face_distance([known_encoding], face_encoding)
            
            # 计算相似度
            # 在获取了距离之后，我们将相似度计算为百分比。由于较低的距离意味着较高的相似度，我们将距离从1中减去。然后，我们乘以100将其转换为百分比。
            #face_distance[0] 访问了 face_distance 数组的第一个（也是唯一的）元素，其中包含计算得到的距离。
            similarity = (1 - face_distance[0]) * 100
            
                
            
            #face_locations 是一个包含检测到的人脸位置信息的列表。每个元素都是一个包含四个整数值的元组，分别代表人脸矩形框的上边界、右边界、下边界和左边界。
            #(top, right, bottom, left) 是一个解包操作，将元组中的四个值依次赋给 top、right、bottom 和 left 这四个变量。这些变量代表了矩形框的四个边界。
            
            for (top, right, bottom, left) in face_locations:
                
                # 在人脸周围绘制矩形框
                #    cv2.rectangle() 是 OpenCV 提供的一个函数，用于在图像上绘制矩形框。
                #frame 是当前视频帧的图像数据，我们要在这个图像上绘制矩形框。
                #(left, top) 和 (right, bottom) 分别是矩形框的左上角和右下角的坐标。
                #(0, 255, 0) 是矩形框的颜色，这里是绿色，其RGB表示为 (0, 255, 0)。
                #2 是矩形框的线宽，表示矩形框的边界线宽度为2个像素。
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # 在矩形框上方显示相似度
                #cv2.putText() 用于在帧上写入文本。
                #在这里，我们在帧上显示相似度百分比。
                #文本显示在 (left, top - 10) 的位置，略高于围绕人脸的矩形的左上角。
                #{similarity:.2f}% 将相似度值格式化为两位小数，后跟一个百分号。
                #cv2.FONT_HERSHEY_SIMPLEX 指定了字体类型。
                #0.5 是字体比例。
                #(0, 255, 0) 表示文本的颜色（以BGR格式，这里是绿色）。
                #2 是文本的粗细。
                cv2.putText(frame, f'Similarity: {similarity:.2f}%', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # 显示图像
        cv2.imshow('Video', frame)

        # 检测到按键 "q" 退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        print("error")

# 释放摄像头资源
video_capture.release()
cv2.destroyAllWindows()
