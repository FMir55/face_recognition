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

    # tracker
    max_distance_between_points = 200 
    initialization_delay = 10

    # capture
    # path_face_db = Path('face_db')
    path_face_db = Path('../face_db')

    # match face
    similarity_thresh = 73 # 0~100
    match_delay = 5
    warmup_delay = 20

    # draw
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255,255,255)
    pivot_img_size = 112 #face recognition result image

    # msg
    msg_no_face = "No face detected(press 'q' to quit)"

    # msg
    msg_face = "face detected(press 'q' to shot)"
    msg_no_face_cap = "No face detected"
    msg_ask_keep = "Wanna keep this face?(y/n)"

    # bpm
    bpm_limits=[50, 160]
