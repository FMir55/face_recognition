from collections import Counter

import cv2
from pycoral.adapters.common import input_size
from pycoral.utils.edgetpu import make_interpreter

from utils.apis import get_attr
from utils.bpm import get_pulse
from utils.config import Args
from utils.draw import draw_identity, make_bpm_plot
from utils.inference import (get_embeddings_v2, inference_detection,
                             inference_embedding)
from utils.preparation import clean_counter, do_identity, get_suspects, prune
from utils.similarity import calc_dist, get_label
from utils.tracker import convert_detection, get_tracker

args = Args()

def main():
    # camera
    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(3, 1920)
    cap.set(4, 1080)

    # face detection
    interpreter_detection = make_interpreter(args.model_detection)
    interpreter_detection.allocate_tensors()
    inference_size_detection = input_size(interpreter_detection)

    # face embedding
    interpreter_emb = make_interpreter(args.model_emb)
    interpreter_emb.allocate_tensors()

    # tracker
    tracker = get_tracker(args.initialization_delay, args.max_distance_between_points)

    # get suspect identities
    suspects = get_suspects(args.path_face_db)

    # get face embeddings
    df = get_embeddings_v2(suspects, interpreter_emb)

    if do_identity(df): id2identity = {}
    id2info, id2cnt, id2bpm = {}, {}, {}
    id2warmup = Counter()
    prev_res = args.msg_no_face
    while cap.isOpened():
        face_names = []
        ret, cv2_im = cap.read()
        if not ret: break

        objs = inference_detection(cv2_im, interpreter_detection, args.threshold)

        height, width, _ = cv2_im.shape
        scale_x, scale_y = width / inference_size_detection[0], height / inference_size_detection[1]
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
            crop_bgr = cv2_im[y0:y1, x0:x1]
            cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # do attribute/identity
            if id2warmup[id] >= args.warmup_delay:
                # attribute
                attr = get_attr(id, id2info, crop_bgr)
                color = (255, 0, 0) if attr.split(',')[0] == 'Male' else (0, 0, 255)

                # identity
                if do_identity(df):
                    # Already extracted
                    if id in id2identity:
                        suspect_name, label, best_similarity = id2identity[id]
                        face_names.append(label)
                        # Identity checked
                        if suspect_name:
                            label += f"_{best_similarity}%"
                            draw_identity(suspect_name, label, cv2_im, (x0, y0, x1, y1), args)

                        # Unknown checked
                        else:
                            attr += f"({label})"

                    # Not yet
                    else:
                        try:
                            df['embedding_sample'] = [inference_embedding(crop_bgr, interpreter_emb)] * len(df)
                            df['distance'] = df.apply(calc_dist, axis = 1)
                            candidate = df.sort_values(by = ["distance"]).iloc[0]
                            suspect_name = candidate['suspect']
                            best_distance = candidate['distance']
                            best_similarity = int((1 - best_distance)* 100)

                            label = get_label(suspect_name) if best_similarity >= args.similarity_thresh else 'Unknown'
                            id2cnt[id][label] += 1
                            print(id, id2cnt[id].most_common(), label, best_similarity)
                            if id2cnt[id][label] >= args.match_delay:
                                id2identity[id] = (suspect_name if label != 'Unknown' else None, 
                                                    label,
                                                    best_similarity)
                                
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
                name = label if do_identity(df) else "Data display"
                plot_title = f"{name} - raw signal (top) and PSD (bottom)"
                if id in id2bpm:
                    text_bpm = id2bpm[id].run(crop_bgr)
                    cv2.putText(cv2_im, text_bpm, (x0, y1+60), args.font, 1.0, color, 2)

                    # data display
                    plot = make_bpm_plot(id2bpm[id], crop_bgr)
                    cv2.imshow(plot_title, plot)
                else:
                    cv2.destroyWindow(plot_title)

                # draw
                cv2.putText(cv2_im, attr, (x0, y1+30), args.font, 1.0, color, 2)

            # 高乘載管制:1
            break
        
        clean_counter(id2warmup, ids)
        if do_identity(df): clean_counter(id2identity, ids)
        clean_counter(id2bpm, ids)

        res = ("No identity detected" if len(face_names)==0 else '_'.join(face_names)) + "(Press 'q' to quit)"
        if res != prev_res: cv2.destroyAllWindows()
        cv2.imshow(res, cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_res = res

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
