import cv2
def face_reveal(net, img, conf_threshold = 0.5):
    copy_img = img.copy()
    imgHeight = img.shape[0]
    imgWidth = img.shape[1]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * imgWidth)
            y1 = int(detections[0, 0, i, 4] * imgHeight)
            x2 = int(detections[0, 0, i, 5] * imgWidth)
            y2 = int(detections[0, 0, i, 6] * imgHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(copy_img, (x1, y1), (x2, y2), (0, 255, 0), int(round(imgHeight/150)), 8)
    return copy_img, faceBoxes
genderProto='gender_deploy.prototxt'
genderModel='gender_net.caffemodel'
genderList = ['Male', 'Female']
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceProto = 'opencv_face_detector.pbtxt'
faceModel = 'opencv_face_detector_uint8.pb'
faceNet = cv2.dnn.readNet(faceModel, faceProto)
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
img = cv2.imread('test_image_5.png')
res, faceBoxes = face_reveal(faceNet, img)
faceCount = 0
maleCount = 0
femaleCount = 0
for faceBox in faceBoxes:
    faceCount = faceCount + 1
    face = img[max(0, faceBox[1] - 15):
                 min(faceBox[3] + 15, img.shape[0] - 1),
           max(0, faceBox[0] - 15):min(faceBox[2] + 15,
                                       img.shape[1] - 1)]
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [104, 117, 123], False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    if gender == 'Male':
        maleCount = maleCount + 1
    elif gender == 'Female':
        femaleCount = femaleCount + 1
    cv2.putText(res, f'{gender}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)
cv2.imshow('face_reveal', res)
print(f"faceCount: {faceCount}")
print(f"maleCount: {maleCount}")
print(f'femaleCount: {femaleCount}')
cv2.waitKey(0)
cv2.destroyAllWindows()
