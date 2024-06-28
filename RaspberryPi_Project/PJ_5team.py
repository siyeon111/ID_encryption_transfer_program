import AES_decrypted as Dec
import AES_encryption as Enc
import Send_id_pwd as Send_id
import Send_user_info
import cv2
import numpy as np
import time
import os
import io
import copy
from google.cloud import vision
import face_recognition
import threading
import mySQL_def as Sql
from mySQL_def import MySQLConnector
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    Response,
    redirect,
    url_for,
    session,
)

num = 0
num2 = 0
state = False


login_userid = ""
fname = ""
fname2 = ""
fnameNum = ""
captured = [0, 0, 0, 0]
image_processing_completed = False
com_name = ""
text_list = []


def OCR(input_name):
    client = vision.ImageAnnotatorClient()
    file_name = os.path.abspath(input_name + ".jpg")

    with io.open(file_name, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # 텍스트 감지
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        # 줄바꿈은 유지하면서 각 줄에서 모든 공백을 제거합니다.
        processed_text = "\n".join(
            [line.replace(" ", "") for line in texts[0].description.split("\n")]
        )

        with open(
            input_name + ".txt",
            "w",
            encoding="ANSI",
        ) as file:
            file.write(processed_text)
    else:
        print("텍스트를 인식할 수 없습니다.")


def find_id_card_contours(contours):
    id_card_contours = []
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:  # 사각형이면
            area = cv2.contourArea(contour)
            if 700 < area < 130000:  # 신분증 크기 조건
                (x, y, w, h) = cv2.boundingRect(approx)
                aspectRatio = w / float(h)
                if 1.15 < aspectRatio < 2.0:  # 신분증 비율 조건
                    id_card_contours.append(contour)
    return id_card_contours


def apply_mosaic_to_id_number(frame, contour):
    global mosiac_state
    mosiac_state = False
    x, y, w, h = cv2.boundingRect(contour)
    # 지장 위치 추정
    seal_x_start = x + int(w * 0.70)
    seal_x_end = x + int(w * 0.85)
    seal_y_start = y + int(h * 0.7)
    seal_y_end = y + int(h * 0.9)

    # 지장 ROI에 사각형 그리기
    # cv2.rectangle(
    #     frame, (seal_x_start, seal_y_start), (seal_x_end, seal_y_end), (255, 0, 0), 2
    # )  # 빨간색으로 지장 위치 표시

    # 지장 ROI
    seal_roi = frame[seal_y_start:seal_y_end, seal_x_start:seal_x_end]

    # HSV로 변환
    hsv_roi = cv2.cvtColor(seal_roi, cv2.COLOR_BGR2HSV)

    # 빨간색의 HSV 범위
    lower_red1 = np.array([0, 30, 20])  # [0, 70, 50]
    upper_red1 = np.array([25, 255, 255])  # [10, 255, 255]
    lower_red2 = np.array([160, 30, 20])  # [170, 70, 50]
    upper_red2 = np.array([180, 255, 255])  # [180, 255, 255]

    # 핑크색 hsv범위
    lower_pink = np.array([150, 30, 20])  # 핑크색 하한
    upper_pink = np.array([180, 255, 255])  # 핑크색 상한

    # 마스크 생성
    mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv_roi, lower_pink, upper_pink)

    mask = cv2.bitwise_or(mask1, mask2, mask3)

    # 빨간색 픽셀의 비율 계산
    red_ratio = cv2.countNonZero(mask) / (seal_roi.shape[0] * seal_roi.shape[1])
    # 빨간색 지장이 충분히 있는지 확인
    if red_ratio > 0.05:  # 빨간색 지장이 충분히 있다면, 신분증이 정방향으로 가정
        x_start = x + int(w * 0.28)
        x_end = x + int(w * 0.5)
        y_start = y + int(h * 0.39)
        y_end = y + int(h * 0.5)
    else:  # 신분증이 거꾸로 있다고 가정
        x_start = x + int(w * 0.5)
        x_end = x + int(w * 0.72)
        y_start = y + int(h * 0.5)
        y_end = y + int(h * 0.62)

    # 모자이크 적용
    roi = frame[y_start:y_end, x_start:x_end]
    roi = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
    roi = cv2.resize(
        roi, (x_end - x_start, y_end - y_start), interpolation=cv2.INTER_NEAREST
    )
    frame[y_start:y_end, x_start:x_end] = (
        roi  # 모자이크 적용된 영역을 원본 이미지에 덮어씌움
    )


def ROI1():  # argc, argv
    # 캠 실행
    global num
    global state
    global fname
    global fname2
    global fnameNum
    global captured
    global image_processing_completed
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("실패")
        return -1

    img_video, img_capt = None, None
    start = time.perf_counter()
    contourFound = False

    while True:
        ret, img_video = cap.read()
        if not ret:
            print("실패 실패 실패")
            break
        if img_video.size == 0:
            print("영상 실패")
            return -1

        height, width = img_video.shape[:2]
        center_x, center_y = int(width * 0.5), int(height * 0.5)

        img_roi = img_video[
            center_y - int(95 * 1.5) : center_y + int(95 * 1.5),
            center_x - int(150 * 1.5) : center_x + int(150 * 1.5),
        ].copy()
        img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        # img_binary = cv2.adaptiveThreshold(
        #     img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        # )
        img_edge = cv2.Canny(img_gray, 30, 140, 3)

        # 주민등록증 사진
        img_gray2 = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.GaussianBlur(img_gray2, (5, 5), 0)
        # img_binary2 = cv2.adaptiveThreshold(
        #     img_gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        # )
        img_edge2 = cv2.Canny(img_gray2, 30, 140, 3)

        # 컨투어
        contours, _ = cv2.findContours(
            img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours2, _ = cv2.findContours(
            img_edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # roi 영역
        roi = (
            center_x - int(150 * 1.5),
            center_y - int(95 * 1.5),
            int(300 * 1.5),
            int(189 * 1.5),
        )

        # 전체 컨투어
        for contour in contours:
            # 신분증 면적 조건 컨투어 ( 최소한의 크기 이상 지정 )
            if cv2.contourArea(contour) > (300 * 189 * 0.75):
                rect = cv2.boundingRect(contour)

                # 가로 세로 비율의 범위 ( 주민등록증 비율 )
                if abs(1.6 - float(rect[2]) / rect[3]) <= 0.1:
                    # 외각선을 제외한 이미지를 저장하기 위해 roi 복사본인 result
                    img_result = img_roi.copy()
                    # 주민등록증에 해당하는 윤곽선 표시
                    cv2.rectangle(img_roi, rect, (0, 255, 0), 2)

                    for contour2 in contours2:
                        if cv2.contourArea(contour2) > (100 * 125 * 0.75):
                            rect2 = cv2.boundingRect(contour2)

                            if abs(0.8 - float(rect2[3]) / rect2[2]) <= 0.2:
                                cv2.rectangle(img_roi, rect2, (255, 0, 0), 2)

                                # 1초마다 한번 캡쳐하는 조건문
                                # contourFound가 flase 일때 ( 주민등록증을 처음으로 찾았을때 )
                                if not contourFound:
                                    start = time.perf_counter()
                                    contourFound = True
                                else:
                                    # 현재시간 시점을 end에 저장
                                    end = time.perf_counter()
                                    # end - start = Yees
                                    Yees = end - start
                                    # Yees의 간격이 1초이상일때 저장
                                    if Yees >= 1:
                                        # bounding box 내부의 이미지만 복사 ( 보다 더 깔끔한 이미지를 위해 )
                                        # rect는 (x, y, w, h) 형태의 튜플이라고 가정
                                        x, y, w, h = rect
                                        captured = img_result[
                                            y : y + h, x : x + w
                                        ].copy()

                                        cap.release()
                                        cv2.destroyAllWindows()
                                        state = True
                                        image_processing_completed = True
                                        return 0
                                        contourFound = False

        # Create a black image of the same size as img_video (excluding the ROI part)
        img_black = np.zeros_like(img_video)

        # Copy the ROI area from img_video to the same position in the black image (add black image)
        x, y, w, h = roi
        img_roi_resized = cv2.resize(img_roi, (w, h))  # w와 h는 대상 영역의 너비와 높이

        # 조정된 img_roi를 img_black에 복사
        img_black[y : y + h, x : x + w] = img_roi_resized

        # Draw a rectangle on top of the ROI area (highlight the ROI area)
        cv2.rectangle(
            img_black,
            (int(center_x - 150 * 1.5), int(center_y - 95 * 1.5)),
            (int(center_x + 150 * 1.5), int(center_y + 95 * 1.5)),
            (0, 0, 255),
            2,
        )

        # Display images for verification (ROI not needed)

        # cv2.imshow("ROI", img_roi)
        # cv2.imshow("Video", img_black)
        # cv2.imshow("test", img_edge)
        img_mosiac = copy.deepcopy(img_black)
        gray = cv2.cvtColor(img_mosiac, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        id_card_contours = find_id_card_contours(contours)
        for contour in id_card_contours:
            apply_mosaic_to_id_number(img_mosiac, contour)

            cv2.drawContours(img_mosiac, [contour], -1, (0, 255, 0), 3)
        # cv2.imshow("asd", img_mosiac)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + cv2.imencode(".jpg", img_mosiac)[1].tobytes()
        )
        # yield (
        #     b"--frame\r\n"
        #     b"Content-Type: image/jpeg\r\n\r\n"
        #     + cv2.imencode(".jpg", img_black)[1].tobytes()
        # )
        if cv2.waitKey(1) == 27:
            break  # Key input


def ROI2():  # argc, argv
    # 캠 실행
    global num2
    global state
    global fname
    global fname2
    global fnameNum
    global captured
    global image_processing_completed
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("실패")
        return -1

    img_video, img_capt = None, None
    start = time.perf_counter()
    contourFound = False

    while True:
        ret, img_video = cap.read()
        if not ret:
            print("실패 실패 실패")
            break
        if img_video.size == 0:
            print("영상 실패")
            return -1

        height, width = img_video.shape[:2]
        center_x, center_y = int(width * 0.5), int(height * 0.5)

        img_roi = img_video[
            center_y - int(95 * 1.5) : center_y + int(95 * 1.5),
            center_x - int(150 * 1.5) : center_x + int(150 * 1.5),
        ].copy()
        img_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        # img_binary = cv2.adaptiveThreshold(
        #     img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        # )
        img_edge = cv2.Canny(
            img_gray, 35, 140, 3
        )  # img_edge = cv2.Canny(img_gray, 35, 140, 3)

        # 주민등록증 사진
        img_gray2 = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
        img_gray2 = cv2.GaussianBlur(img_gray2, (5, 5), 0)
        # img_binary2 = cv2.adaptiveThreshold(
        #     img_gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        # )
        img_edge2 = cv2.Canny(img_gray, 35, 140, 3)

        # 컨투어
        contours, _ = cv2.findContours(
            img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours2, _ = cv2.findContours(
            img_edge2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # roi 영역
        roi = (
            center_x - int(150 * 1.5),
            center_y - int(95 * 1.5),
            int(300 * 1.5),
            int(189 * 1.5),
        )

        # 전체 컨투어
        for contour in contours:
            # 신분증 면적 조건 컨투어 ( 최소한의 크기 이상 지정 )
            if cv2.contourArea(contour) > (300 * 189 * 0.75):
                rect = cv2.boundingRect(contour)

                # 가로 세로 비율의 범위 ( 주민등록증 비율 )
                if abs(1.6 - float(rect[2]) / rect[3]) <= 0.1:
                    # 외각선을 제외한 이미지를 저장하기 위해 roi 복사본인 result
                    img_result = img_roi.copy()
                    # 주민등록증에 해당하는 윤곽선 표시
                    cv2.rectangle(img_roi, rect, (0, 255, 0), 2)

                    for contour2 in contours2:
                        if cv2.contourArea(contour2) > (100 * 125 * 0.75):
                            rect2 = cv2.boundingRect(contour2)

                            if abs(0.8 - float(rect2[3]) / rect2[2]) <= 0.2:
                                cv2.rectangle(img_roi, rect2, (255, 0, 0), 2)

                                # 1초마다 한번 캡쳐하는 조건문
                                # contourFound가 flase 일때 ( 주민등록증을 처음으로 찾았을때 )
                                if not contourFound:
                                    start = time.perf_counter()
                                    contourFound = True
                                else:
                                    # 현재시간 시점을 end에 저장
                                    end = time.perf_counter()
                                    # end - start = Yees
                                    Yees = end - start
                                    # Yees의 간격이 1초이상일때 저장
                                    if Yees >= 1:
                                        # bounding box 내부의 이미지만 복사 ( 보다 더 깔끔한 이미지를 위해 )
                                        # rect는 (x, y, w, h) 형태의 튜플이라고 가정
                                        x, y, w, h = rect
                                        captured = img_result[
                                            y : y + h, x : x + w
                                        ].copy()

                                        num2 += 1
                                        cap.release()
                                        cv2.destroyAllWindows()
                                        # compare_jpg_txt()
                                        return -1
                                        contourFound = False

        # Create a black image of the same size as img_video (excluding the ROI part)
        img_black = np.zeros_like(img_video)

        # Copy the ROI area from img_video to the same position in the black image (add black image)
        x, y, w, h = roi
        img_roi_resized = cv2.resize(img_roi, (w, h))  # w와 h는 대상 영역의 너비와 높이

        # 조정된 img_roi를 img_black에 복사
        img_black[y : y + h, x : x + w] = img_roi_resized

        # Draw a rectangle on top of the ROI area (highlight the ROI area)
        cv2.rectangle(
            img_black,
            (int(center_x - 150 * 1.5), int(center_y - 95 * 1.5)),
            (int(center_x + 150 * 1.5), int(center_y + 95 * 1.5)),
            (0, 0, 255),
            2,
        )

        img_mosiac = copy.deepcopy(img_black)
        gray = cv2.cvtColor(img_mosiac, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        id_card_contours = find_id_card_contours(contours)
        for contour in id_card_contours:
            apply_mosaic_to_id_number(img_mosiac, contour)

            cv2.drawContours(img_mosiac, [contour], -1, (0, 255, 0), 3)
        # cv2.imshow("asd", img_mosiac)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + cv2.imencode(".jpg", img_mosiac)[1].tobytes()
        )
        # Display images for verification (ROI not needed)
        # yield (
        #     b"--frame\r\n"
        #     b"Content-Type: image/jpeg\r\n\r\n"
        #     + cv2.imencode(".jpg", img_black)[1].tobytes()
        #     + b"\r\n"
        # )

        if cv2.waitKey(1) == 27:
            break  # Key input


def get_image_path(relative_path):
    base_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 파일의 디렉토리 경로
    return os.path.join(base_dir, relative_path)


def process_images(filename1, filename2):
    img1 = get_image_path(filename1)
    img2 = get_image_path(filename2)

    image1 = face_recognition.load_image_file(img1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    face_locations1 = face_recognition.face_locations(image1)
    if not face_locations1:
        print("No faces detected in the first image.")
        return False, "No faces detected"
    setROI1 = face_locations1[0]
    encode1 = face_recognition.face_encodings(image1)[0]
    ROI1 = cv2.rectangle(
        image1, (setROI1[3], setROI1[0]), (setROI1[1], setROI1[2]), (255, 0, 0), 2
    )

    image2 = face_recognition.load_image_file(img2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    face_locations2 = face_recognition.face_locations(image2)
    if not face_locations2:
        print("No faces detected in the second image.")
        return False, "No faces detected"
    setROI2 = face_locations2[0]
    encode2 = face_recognition.face_encodings(image2)[0]
    ROI2 = cv2.rectangle(
        image2, (setROI2[3], setROI2[0]), (setROI2[1], setROI2[2]), (255, 0, 0), 2
    )

    # f"{number:.2f}"
    Same = face_recognition.compare_faces([encode1], encode2)
    difference = face_recognition.face_distance([encode1], encode2)
    # difference = f"{(1-difference)*100:.2f}"
    # difference = str(difference).strip()
    difference = f"{((1 - difference[0]) * 100):.2f}"
    difference = str(difference).strip()
    print(Same)

    return Same[0], difference + "%"

    # cv2.imshow("image1", ROI1)
    # cv2.imshow("image2", ROI2)

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def compare_jpg_txt(id):
    global text_list
    compare_jpg = False
    compare_txt = False

    # DB에서 기존 저장된 파일 가져와서 복호화, TXT, JPG
    Dec.decrypt_user_data(id, "decrypted_file/")

    # 기존 파일(ROI1) / 비교할 새 파일(ROI2) 가져오기
    file1_jpg = "decrypted_file/" + id + ".jpg"
    file1_txt = "decrypted_file/" + id + ".txt"
    file2_jpg = "compare_file/" + id + ".jpg"
    file2_txt = "compare_file/" + id + ".txt"

    # Face_recognition 적용하기 및 이미지 비교
    compare_jpg, accuracy = process_images(file1_jpg, file2_jpg)
    if accuracy == "No faces detected":
        text_list.append("No faces detected!!")
        text_list.append("Try Again.")
        return
    if compare_jpg:
        text_list.append("이미지 비교 결과 : 일치")
        text_list.append("이미지 일치율 : {0}\n".format(accuracy))
    elif not compare_jpg:
        text_list.append("이미지 비교 결과 : 불일치")
        text_list.append("이미지 일치율 : {0}\n".format(accuracy))

    # 텍스트 비교 (파일 열어서 3줄까지만 가져와서 비교)
    lines1, lines2 = [], []
    input_file1 = open(file1_txt, "r")
    input_file2 = open(file2_txt, "r")
    test_text1 = ""
    test_text2 = ""

    # 예외처리(호오오옥시나 못 열까봐)
    if not input_file1:
        text_list.append(
            "database1 폴더에서",
            file1_txt,
            ".txt 파일을 열 수 없습니다.",
        )
        return
    if not input_file2:
        text_list.append(
            "database2 폴더에서",
            file1_txt,
            ".txt 파일을 열 수 없습니다.",
        )
        return

    # 연 파일에서 텍스트 읽어와서 텍스트 한 줄씩을 벡터의 한 요소씩에 담기
    for line in input_file1:
        lines1.append(line)
    for line in input_file2:
        lines2.append(line)

    # 요건 이름에서 한자빼려고 하는거임
    lines1[1] = lines1[1][: lines1[1].find("(")]
    lines2[1] = lines2[1][: lines2[1].find("(")]

    # 텍스트를 (3줄까지)변수로 저장. 여기서 줄바꿈 문자까지 있다!
    for i in range(3):
        test_text1 += lines1[i]
        test_text1 += "\n"
        test_text2 += lines2[i]
        test_text2 += "\n"

    # 두 파일에서 따온 텍스트 비교
    if test_text1 == test_text2:
        text_list.append("텍스트 매칭 : 일치\n")
        # print("텍스트 : 일치")
        compare_txt = True
        # print(".txt파일의 텍스트가 일치합니다.")
    if test_text1 != test_text2:
        text_list.append("텍스트 매칭 : 불일치\n")
        # print("텍스트 : 불일치")
        compare_txt = False

    input_file1.close()
    input_file2.close()

    text_list.append("신분증 판별이 완료되었습니다!")

    if (compare_jpg == True) and (compare_txt == True):
        text_list.append("신분증 매칭 성공!")

    elif (compare_jpg == False) or (compare_txt == False):
        text_list.append("신분증 매칭 실패")
    else:
        text_list.append("오류")


app = Flask(__name__)

app.secret_key = "@asd8923njfgw9%$@gds3"
# app.secret_key = "@asd8923njfgw9%$@gds3"


@app.route("/")
def home():
    return render_template("login.html")


@app.route("/login", methods=["POST"])
def login():
    global login_userid
    user_id = request.form["user_id"]
    password = request.form["password"]

    # MySQL 데이터베이스 연결
    connector = MySQLConnector("192.168.0.62", "user1", "1234", "identification")
    connector.connect()

    # 데이터베이스에서 사용자 정보 조회
    query = "SELECT * FROM id_pwd WHERE id = %s AND password = %s"
    connector.cursor.execute(query, (user_id, password))
    account = connector.cursor.fetchone()

    # 데이터베이스 연결 해제
    connector.disconnect()

    if account:
        login_userid = user_id
        return redirect(url_for("mainmenu"))  # 로그인 성공 시 메인 메뉴로 리디렉션
    else:
        return "로그인 실패! 정보를 확인해 주세요.", 401  # 로그인 실패 시 메시지 출력


@app.route("/mainmenu")
def mainmenu():
    # '/mainmenu' 페이지를 위한 로직을 여기에 추가합니다.
    return render_template("index.html")  # 'index.html' 페이지를 렌더링합니다.


@app.route("/signup")
def signup_form():
    return render_template("sign-up.html")


@app.route("/button2")
def button2():
    # 버튼 2 클릭 시, session에 클릭 여부를 저장합니다.
    session["button2_clicked"] = True
    return redirect(url_for("home"))


@app.route("/signup", methods=["POST"])
def signup():
    if "userid_checked" not in session or not session["userid_checked"]:
        # 중복 확인이 이루어지지 않은 경우
        return redirect(url_for("signup_form", alert=True))
    if session["is_duplicate"]:
        # 중복된 아이디인 경우
        return redirect(url_for("signup_form", alert=True))

    # 회원가입 처리 로직
    username = request.form["username"]
    userid = request.form["userid"]
    password = request.form["password"]
    phone = request.form["phone"]
    Send_id.send_id_pwd(userid, password, phone)  # 실제 회원가입 처리 코드
    session["userid_checked"] = False  # 세션 초기화
    return redirect(url_for("home"))  # 회원가입 후 홈으로 리다이렉트


@app.route("/check_userid", methods=["POST"])
def check_userid():
    userid = request.form["userid"]
    # 여기에 userid가 데이터베이스에 존재하는지 확인하는 로직 구현
    # 예를 들어, 데이터베이스 조회 결과를 is_duplicate 변수에 할당
    is_duplicate = Sql.use_check_id_exists(userid)  # 중복확인 함수 실제 구현 필요

    session["userid_checked"] = True
    session["is_duplicate"] = is_duplicate

    return jsonify({"is_duplicate": is_duplicate})


@app.route("/hello", methods=["GET", "POST"])
def hello_world():
    return render_template("hello.html")


@app.route("/image_save", methods=["GET", "POST"])
def image_save():
    global num, image_processing_completed
    if request.method == "POST":
        # 사용자가 입력한 파일 이름 받기

        fname1 = "information_file/" + login_userid + ".jpg"
        fname2 = "information_file/" + login_userid
        cv2.imwrite(fname1, captured)
        OCR(fname2)
        ###### 암호화 이후, 원본 파일 삭제해야함!!! 추후에 처리할 것. ######
        Enc.make_encrypted_file(login_userid + ".jpg")
        Enc.make_encrypted_file(login_userid + ".txt")

        Send_user_info.send_user_info(login_userid)

        num += 1
        image_processing_completed = False
        return jsonify({"success": True, "message": "이미지가 저장되었습니다."})
    else:
        # POST 메서드로 요청이 들어오지 않은 경우
        return "잘못된 요청입니다."


@app.route("/check_status")
def check_status():
    # 이미지 처리 상태를 반환
    global image_processing_completed
    return jsonify({"completed": image_processing_completed})


@app.route("/video_feeds")
def video_feed():
    return Response(ROI1(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/video_feeds2")
def video_feed2():
    return Response(ROI2(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/hi", methods=["GET", "POST"])
def hi_world():
    global captured
    global login_userid
    text_list.clear()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "submit":
            if login_userid:
                fname1 = "compare_file/" + login_userid + ".jpg"
                fname2 = "compare_file/" + login_userid
                cv2.imwrite(fname1, captured)
                OCR(fname2)

                # 성공 응답 보내기
                return jsonify(
                    {
                        "message": "제출이 성공적으로 처리되었습니다.",
                        "filename": login_userid,
                    }
                )
            else:
                # 오류 응답 보내기
                return jsonify({"error": "파일 이름이 제공되지 않았습니다."}), 400
        elif action == "compare":
            text_list.clear()
            compare_jpg_txt(login_userid)

            # return render_template("hi.html", messages=text_list)
            return jsonify({"messages": text_list})

    # GET 요청 처리
    return render_template("hi.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5009, debug=True, threaded=True)
