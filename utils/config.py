import glob
from pathlib import Path

import cv2


class Args:
    # camera device index
    camera_idx = 2

    # face detection
    # model = 'all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    model_detection = '../all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    threshold = 0.1

    # face embedding
    # model_emb = 'all_models/facenet.tflite'
    model_emb = '../all_models/facenet.tflite'

    # face_attribute
    model_emotion = '../all_models/emotion.tflite'

    # tracker
    max_distance_between_points = 300 
    initialization_delay = 10

    # capture
    # path_face_db = Path('face_db')
    path_face_db = Path('../face_db')

    # match face
    similarity_thresh = 73 # 0~100
    match_delay = 5
    warmup_delay = 20

    # msg
    msg_no_face = "No face detected(press 'q' to quit)"

    # msg
    msg_face = "face detected(press 'q' to shoot)"
    msg_no_face_cap = "No face detected"
    msg_ask_keep = "Wanna keep this face?(y/n)"

    # emotion
    emotion_table = {
        "Angry":"生氣",
        "Disgust":"作嘔",
        "Fear":"恐懼",
        "Happy":"喜悅",
        "Sad":"傷心",
        "Surprise":"驚訝",
        "Neutral":"正常"
    }

    # bpm
    bpm_limits=[50, 160]

    # draw
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255,255,255)
    scale = 1.0
    textSize = 35
    textSize_bpm = 25
    thickness = 3
    scale_x = 0.1
    scale_y = 1.0
    path_font = 'fonts/simsun.ttc'

    pivot_img_size = 112 #face recognition result image

    scene_height = 1000
    plot_title = "(Press 'q' to quit)"
    face_table = {
        suspect_name:cv2.imread(suspect_name)
        for folder in glob.glob(str(path_face_db / Path('*')))\
        for suspect_name in glob.glob(folder + '/*')
    }

