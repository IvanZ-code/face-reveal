import cv2
import streamlit as st
import numpy as np


def face_reveal(net, img, conf_threshold=0.2):
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
            cv2.rectangle(copy_img, (x1, y1), (x2, y2), (0, 255, 0), int(round(imgHeight / 150)), 8)
    return copy_img, faceBoxes


genderProto='.data/gender_detector/gender_deploy.prototxt'
genderModel='.data/gender_detector/gender_net.caffemodel'
genderList = ['Male', 'Female']
genderNet = cv2.dnn.readNet(genderModel, genderProto)
faceProto = '.data/face_detector/opencv_face_detector.pbtxt'
faceModel = '.data/face_detector/opencv_face_detector_uint8.pb'
faceNet = cv2.dnn.readNet(faceModel, faceProto)
MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)


upload_img = st.file_uploader("Choose your file")

def work_with_image(img):
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
        cv2.putText(res, f'{gender}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1,
                    cv2.LINE_AA)
    return res, faceCount, maleCount, femaleCount


def show_result(img, res, faceCount, maleCount, femaleCount):
    st.image(img, caption='image', channels='BGR')
    st.write('face count: ', faceCount)
    st.write('male count: ', maleCount)
    st.write('female count: ', femaleCount)
    st.image(res, caption='face_reveal', channels='BGR')


if upload_img is None:
    test_img_num = 0
    st.write('Click to see result of some test images')
    t1, t2, t3, t4, t5, t6 = st.columns(6)
    with t1:
        if st.button('test 1'):
            test_img_num = 1
    with t2:
        if st.button('test 2'):
            test_img_num = 2
    with t3:
        if st.button('test 3'):
            test_img_num = 3
    with t4:
        if st.button('test 4'):
            test_img_num = 4
    with t5:
        if st.button('test 5'):
            test_img_num = 5
    with t6:
        if st.button('test 6'):
            test_img_num = 6
    if test_img_num == 1:
        test_img = cv2.imread('.test/test_image_1.png')
        test_res, test_faceCount, test_maleCount, test_femaleCount = work_with_image(test_img)
        show_result(test_img, test_res, test_faceCount, test_maleCount, test_femaleCount)
    elif test_img_num == 2:
        test_img = cv2.imread('.test/test_image_2.png')
        test_res, test_faceCount, test_maleCount, test_femaleCount = work_with_image(test_img)
        show_result(test_img, test_res, test_faceCount, test_maleCount, test_femaleCount)
    elif test_img_num == 3:
        test_img = cv2.imread('.test/test_image_3.png')
        test_res, test_faceCount, test_maleCount, test_femaleCount = work_with_image(test_img)
        show_result(test_img, test_res, test_faceCount, test_maleCount, test_femaleCount)
    elif test_img_num == 4:
        test_img = cv2.imread('.test/test_image_4.png')
        test_res, test_faceCount, test_maleCount, test_femaleCount = work_with_image(test_img)
        show_result(test_img, test_res, test_faceCount, test_maleCount, test_femaleCount)
    elif test_img_num == 5:
        test_img = cv2.imread('.test/test_image_5.png')
        test_res, test_faceCount, test_maleCount, test_femaleCount = work_with_image(test_img)
        show_result(test_img, test_res, test_faceCount, test_maleCount, test_femaleCount)
    elif test_img_num == 6:
        test_img = cv2.imread('.test/test_image_6.png')
        test_res, test_faceCount, test_maleCount, test_femaleCount = work_with_image(test_img)
        show_result(test_img, test_res, test_faceCount, test_maleCount, test_femaleCount)

if upload_img is not None:
    file_bytes = np.asarray(bytearray(upload_img.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    st.image(opencv_image, caption = 'image', channels = 'BGR')
    img = opencv_image
    res, faceCount, maleCount, femaleCount = work_with_image(img)
    st.write('face count: ', faceCount)
    st.write('male count: ', maleCount)
    st.write('female count: ', femaleCount)
    if st.button('face reveal'):
        st.image(res, caption = 'face_reveal', channels = 'BGR')

#cv2.imshow('face_reveal', res)
#print(f"faceCount: {faceCount}")
#print(f"maleCount: {maleCount}")
#print(f'femaleCount: {femaleCount}')
#cv2.waitKey(0)
#cv2.destroyAllWindows()
