from collections import Counter

import cv2
import numpy as np

from utils.bpm import get_pulse
from utils.config import Args
from utils.debug import match_info
from utils.draw import clean_plot, draw_attr, draw_bpm, draw_identity_v2
from utils.inference_v2 import (get_attr, get_embeddings_v2,
                                inference_detection, match)
from utils.preparation import clean_counter, do_identity, get_suspects, prune
from utils.similarity import get_label
from utils.tracker import convert_detection, get_tracker

args = Args()

def main():
    # camera
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(3, 5000)
    cap.set(4, 5000)

    # tracker
    tracker = get_tracker(args.initialization_delay, args.max_distance_between_points)

    # get suspect identities
    suspects = get_suspects(args.path_face_db)

    # get face embeddings
    df = get_embeddings_v2(suspects)

    if do_identity(df): id2identity = {}
    id2info, id2cnt, id2bpm = {}, {}, {}
    id2warmup = Counter()
    while cap.isOpened():
        # (980 1280 3)
        ret, cv2_im = cap.read()
        
        cv2_clean = cv2_im.copy()
        h, w, _ = cv2_im.shape
        w_new = int(w/h*args.scene_height)
        info_box = np.ones(
            (args.scene_height, 1920-w_new, 3),
            dtype=np.uint8
        )*255
        if not ret: break

        objs, scale_x, scale_y = inference_detection(cv2_im, args.threshold)
        detections = [convert_detection(obj, scale_x, scale_y) for obj in objs]
        tracked_objects = tracker.update(detections=detections)
        ids = []
        for tracked_object in tracked_objects:
            id = tracked_object.id
            ids.append(id)
            if id not in id2cnt: 
                id2cnt[id] = Counter()
            id2warmup[id] += 1

            x0, y0, x1, y1 = tracked_object.last_detection.points.flatten()
            x0, y0, x1, y1 = prune(x0, y0, x1, y1)
            crop_bgr = cv2_clean[y0:y1, x0:x1]
            cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # do attribute/identity
            if id2warmup[id] >= args.warmup_delay:
                # attribute
                gender, age, emotion = get_attr(id, id2info, crop_bgr)
                color = (0, 0, 255) if gender.startswith('Male') else (255, 0, 0)
                draw_attr(info_box, gender, age, emotion, color)

                # identity
                if do_identity(df):
                    # Already extracted
                    if id in id2identity:
                        suspect_name, label, best_similarity = id2identity[id]
                        # Identity checked
                        if suspect_name:
                            label += f"_{best_similarity}%"
                        # Unknown checked
                        else:
                            pass

                        draw_identity_v2(info_box, suspect_name, label, color)

                    # Not yet
                    else:
                        try:
                            # match identity
                            suspect_name, best_similarity = match(df, crop_bgr)
                            label = get_label(suspect_name) if best_similarity >= args.similarity_thresh else f"Unknown{id}"
                            id2cnt[id][label] += 1

                            match_info(id, label, best_similarity, id2cnt)
                            
                            if id2cnt[id][label] >= args.match_delay:
                                id2identity[id] = (suspect_name if not label.startswith('Unknown') else None, 
                                                    label, best_similarity)
                                
                                # Start bpm
                                id2bpm[id] = get_pulse(args.bpm_limits)

                                # End the count
                                del id2cnt[id]
                        except Exception as err:
                            print(str(err))

                # without identity
                else:
                    # attribute complete & bpm not started
                    if id in id2info and id not in id2bpm:
                        # start bpm
                        id2bpm[id] = get_pulse(args.bpm_limits)

                # run bpm
                if id in id2bpm:
                    text_bpm = id2bpm[id].run(crop_bgr)
                    draw_bpm(info_box, crop_bgr, text_bpm, id2bpm[id], color)

            # 高乘載管制:1
            break
        
        clean_counter(id2warmup, ids)
        if do_identity(df): clean_counter(id2identity, ids)
        clean_plot(id2bpm, ids)
        clean_counter(id2bpm, ids)

        # resize
        cv2_im = cv2.resize(
            cv2_im, 
            (w_new, args.scene_height), 
            interpolation=cv2.INTER_AREA
        )
        # concat (1080, 1920, 3)
        cv2_final = cv2.hconcat([info_box, cv2_im])
        cv2.imshow(args.plot_title, cv2_final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
