from collections import Counter
from itertools import cycle

import cv2

from utils.config import Args
from utils.draw_v2 import (draw_attr, draw_bpm, draw_identity,
                           get_default_info_box, put_default_text)
from utils.get_info_v2 import (do_identity, get_age_gender, get_bpm_emotion,
                               get_identity)
from utils.inference_v3 import inference_detection
from utils.preparation_v2 import clean_dict, prune
from utils.tracker import convert_detection, get_tracker

args = Args()

c_emotion = cycle(range(args.delay_emotion))

def main():
    # camera
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(3, 5000)
    cap.set(4, 5000)

    # tracker
    tracker = get_tracker(args.initialization_delay, args.max_distance_between_points)

    if do_identity(): id2identity = {}
    id2info, id2emotion, id2bpm = {}, {}, {}
    id2warmup = Counter()
    while cap.isOpened():
        # (980 1280 3)
        ret, cv2_im = cap.read()
        
        cv2_clean = cv2_im.copy()
        h, w, _ = cv2_im.shape
        w_new = int(w/h*args.scene_height)

        info_box = get_default_info_box(w_new)
        if not ret: break

        objs, scale_x, scale_y = inference_detection(cv2_im, args.threshold)
        detections = [convert_detection(obj, scale_x, scale_y) for obj in objs]
        tracked_objects = tracker.update(detections=detections)
        ids = []
        for tracked_object in tracked_objects:
            id = tracked_object.id
            ids.append(id)
            id2warmup[id] += 1

            x0, y0, x1, y1 = tracked_object.last_detection.points.flatten()
            x0, y0, x1, y1 = prune(x0, y0, x1, y1)
            crop_bgr = cv2_clean[y0:y1, x0:x1]
            cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # do attribute/identity
            if id2warmup[id] >= args.warmup_delay:
                # attribute
                get_age_gender(id, id2info, crop_bgr)
                age, gender = id2info[id]['age'], id2info[id]['gender']
                color = (0, 0, 0) if not gender else\
                        (0, 0, 255) if gender.startswith('Male') else (255, 0, 0)
                # draw
                if gender:
                    info_box = draw_attr(info_box, gender, color, 1)
                if age:
                    info_box = draw_attr(info_box, age, color, 2)

                # bpm & emotion
                get_bpm_emotion(id, id2bpm, id2emotion, crop_bgr, next(c_emotion))
                text_bpm = id2bpm[id].text
                emotion = id2emotion[id]
                # draw
                info_box = draw_bpm(info_box, crop_bgr, text_bpm, id2bpm[id], color)
                info_box = draw_attr(info_box, emotion, color, 3)
                
                # identity
                if do_identity():
                    get_identity(id, id2identity, crop_bgr)
                    (suspect_name, label), _ = id2identity[id].most_common(1)[0] \
                        if id in id2identity and len(id2identity[id]) != 0 else \
                        ((None, f"Guest{id}"), 0)

                    # debug
                    label += f"_({len(id2identity[id]) if id in id2identity else None})"
                    
                    info_box = draw_identity(info_box, suspect_name, label, color)

            # 高乘載管制:1
            break
        
        clean_dict(id2warmup, ids)
        if do_identity(): clean_dict(id2identity, ids)
        clean_dict(id2bpm, ids)
        clean_dict(id2emotion, ids)

        if (info_box == get_default_info_box(w_new)).all():
            info_box = put_default_text(info_box)

        # resize
        cv2_im = cv2.resize(
            cv2_im, 
            (w_new, args.scene_height), 
            interpolation=cv2.INTER_AREA
        )
        # concat (1080, 1920, 3)
        cv2_final = cv2.hconcat([info_box, cv2_im])

        cv2.namedWindow(args.plot_title, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(args.plot_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(args.plot_title, cv2_final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
