from collections import Counter
from pathlib import Path

import cv2
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference

from utils.apis import get_attr, get_embedding, get_embeddings
from utils.draw import draw_identity
from utils.preparation import clean_counter, get_suspects, prune
from utils.similarity import findDistance, get_label
from utils.tracker import convert_detection, get_tracker


class Args:
    # camera device index
    camera_idx = 2

    # face detection
    model = 'all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    threshold = 0.1

    # tracker
    max_distance_between_points = 200 
    initialization_delay = 10

    # capture
    path_face_db = Path('face_db')

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

args = Args()

def main():
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

    # get suspect identities
    suspects = get_suspects(args.path_face_db)

    # get face embeddings
    df = get_embeddings(suspects)

    id2info = {}
    if df is not None and df.shape[0] > 0: id2identity = {}
    id2cnt = {}
    id2warmup = Counter()
    prev_res = args.msg_no_face
    while cap.isOpened():
        face_names = []
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
        ids = []
        for tracked_object in tracked_objects:
            id = tracked_object.id
            ids.append(id)
            if id not in id2cnt: id2cnt[id] = Counter()
            id2warmup[id] += 1

            x0, y0, x1, y1 = tracked_object.last_detection.points.flatten()
            x0, y0, x1, y1 = prune(x0, y0, x1, y1)
            crop_bgr = cv2_im[y0:y1, x0:x1]
            cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)

            if id2warmup[id] >= args.warmup_delay:
                attr = get_attr(id2info, crop_bgr)

                # At least one template exists
                if df is not None and df.shape[0] > 0:
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
                            df['embedding_sample'] = [get_embedding(crop_bgr)] * len(df)
                            df['distance'] = df.apply(findDistance, axis = 1)
                            candidate = df.sort_values(by = ["distance"]).iloc[0]
                            suspect_name = candidate['suspect']
                            best_distance = candidate['distance']
                            best_similarity = int((1 - best_distance)* 100)

                            label = get_label(suspect_name) if best_similarity >= args.similarity_thresh else 'Unknown'
                            id2cnt[id][label] += 1
                            print(id, label, best_similarity, id2cnt[id].most_common())
                            if id2cnt[id][label] >= args.match_delay:
                                id2identity[id] = (suspect_name if label != 'Unknown' else None, 
                                                    label,
                                                    best_similarity)
                                del id2cnt[id]
                        except Exception as err:
                            print(str(err))
                    
                    clean_counter(id2identity, ids)

                # draw
                cv2.putText(cv2_im, attr, (x0, y0+30), args.font, 1.0, (255, 0, 0), 2)

            # 高乘載管制:1
            break
        
        clean_counter(id2warmup, ids)

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
