import time
from pathlib import Path

import cv2
import numpy as np
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference

from utils.preparation import get_legal_fname, mkdir
from utils.tracker import convert_detection, get_tracker


class Args:
    # camera device index
    camera_idx = 2

    # face detection
    model = 'all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    threshold = 0.1

    # tracker
    max_distance_between_points = 100 
    initialization_delay = 10

    # capture
    path_face_db = Path('face_db')

    # msg
    msg_face = "face detected(press 'q' to shot)"
    msg_no_face = "No face detected"
    msg_ask_keep = "Wanna keep this face?(y/n)"

args = Args()

def main():
    identity = input('Who are you...?\n')
    mkdir(args.path_face_db)

    # camera
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(3, 1920)
    cap.set(4, 1080)

    # face detection
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    # tracker
    tracker = get_tracker(args.initialization_delay, args.max_distance_between_points)
    prev_res = args.msg_no_face
    while cap.isOpened():
        ret, cv2_im = cap.read()
        if not ret: break

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)

        height, width, _ = cv2_im.shape
        scale_x, scale_y = width / inference_size[0], height / inference_size[1]
        detections = [convert_detection(obj, scale_x, scale_y) for obj in objs]
        tracked_objects = tracker.update(detections=detections)
        for tracked_object in tracked_objects:
            x0, y0, x1, y1 = tracked_object.last_detection.points.flatten()
            crop_bgr = cv2_im[y0:y1, x0:x1]
            
            # draw
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            break

        res = args.msg_no_face if len(tracked_objects) == 0 else args.msg_face
        if res != prev_res: cv2.destroyAllWindows()
        cv2.imshow(res, cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q') and res == args.msg_face:
            print(f'Capturing identity: {identity}')
            cv2.destroyAllWindows()
            cv2.imshow(args.msg_ask_keep, cv2_im)
            if cv2.waitKey(0) & 0xFF == ord('y'):
                path_identity = args.path_face_db / Path(identity)
                mkdir(path_identity)
                fname = get_legal_fname(path_identity, identity)
                # save image
                cv2.imwrite(str(fname), crop_bgr)
                print(f"Face template built: {str(fname)}")
                break
            elif cv2.waitKey(0) & 0xFF == ord('n'):
                print('Face template dropped')
                cv2.destroyAllWindows()
        if res != prev_res: cv2.destroyAllWindows()
        prev_res = res
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
