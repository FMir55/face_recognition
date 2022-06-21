import time
from pathlib import Path

import cv2
import numpy as np
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference

from utils.config import Args
from utils.preparation import get_legal_fname, mkdir, prune
from utils.tracker import convert_detection, get_tracker

args = Args()

def main():
    identity = input('Who are you...?\n')
    mkdir(args.path_face_db)

    # camera
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(3, 1920)
    cap.set(4, 1080)

    # face detection
    interpreter_detection = make_interpreter(args.model_detection)
    interpreter_detection.allocate_tensors()
    inference_size_detection = input_size(interpreter_detection)

    # tracker
    tracker = get_tracker(args.initialization_delay, args.max_distance_between_points)
    prev_res = args.msg_no_face_cap
    while cap.isOpened():
        ret, cv2_im = cap.read()
        if not ret: break

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size_detection)
        run_inference(interpreter_detection, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter_detection, args.threshold)

        height, width, _ = cv2_im.shape
        scale_x, scale_y = width / inference_size_detection[0], height / inference_size_detection[1]
        detections = [convert_detection(obj, scale_x, scale_y) for obj in objs]
        tracked_objects = tracker.update(detections=detections)
        for tracked_object in tracked_objects:
            x0, y0, x1, y1 = tracked_object.last_detection.points.flatten()
            x0, y0, x1, y1 = prune(x0, y0, x1, y1)
            crop_bgr = cv2_im[y0:y1, x0:x1]
            
            # draw
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            break

        res = args.msg_no_face_cap if len(tracked_objects) == 0 else args.msg_face
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
        prev_res = res
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
