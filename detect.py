from collections import Counter
from pathlib import Path

import cv2
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference

from utils.apis import get_embedding, get_embeddings, get_face_info
from utils.preparation import get_suspects
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
    similarity_thresh = 0.7
    match_delay = 5

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
    prev_res = args.msg_no_face
    while cap.isOpened():
        face_names = []
        ret, cv2_im = cap.read()
        if not ret: break
        resolution_y, resolution_x = cv2_im.shape[:2]

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
            x0, y0, x1, y1 = tracked_object.last_detection.points.flatten()
            crop_bgr = cv2_im[y0:y1, x0:x1]

            if id in id2info: 
                emotion, age, gender = id2info[id].values()
            else:
                try:
                    face_info = get_face_info(crop_bgr)
                    emotion, age, gender = face_info.values()
                    id2info[id] = {
                        "emotion" : emotion, 
                        "age" : age, 
                        "gender" : gender
                    }
                except Exception as err:
                    print(str(err))
                    emotion, age, gender = '', '', ''
            # draw
            attr = f"{gender}, {age}y, {emotion}"
            cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2.putText(cv2_im, attr, (x0, y0+30), args.font, 1.0, (255, 0, 0), 2)

            # At least one template exists
            if df is not None and df.shape[0] > 0:
                if id in id2identity:
                    suspect_name, label, best_similarity = id2identity[id]

                    display_img = cv2.imread(suspect_name)
                    display_img = cv2.resize(display_img, (args.pivot_img_size, args.pivot_img_size))

                    face_names.append(label)
                    label += f"_{best_similarity}%"

                    # draw
                    try:
                        w = x1-x0
                        if y0 - args.pivot_img_size > 0 and x1 + args.pivot_img_size < resolution_x:
                            #top right
                            cv2_im[y0 - args.pivot_img_size:y0, x1:x1+args.pivot_img_size, :3] = display_img

                            overlay = cv2_im.copy(); opacity = 0.4
                            cv2.rectangle(cv2_im,(x1,y0),(x1+args.pivot_img_size, y0+20),(46,200,255),cv2.FILLED)
                            cv2.addWeighted(overlay, opacity, cv2_im, 1 - opacity, 0, cv2_im)
                            cv2.putText(cv2_im, label, (x1, y0+10), args.font, 0.5, args.text_color, 1)

                            #connect face and text
                            cv2.line(cv2_im,(int((x0+x1)/2), y0), (x0+3*int((x1-x0)/4), y0-int(args.pivot_img_size/2)),(67,67,67),1)
                            cv2.line(cv2_im, (x0+3*int((x1-x0)/4), y0-int(args.pivot_img_size/2)), (x1, y0 - int(args.pivot_img_size/2)), (67,67,67),1)

                        elif y1 + args.pivot_img_size < resolution_y and x0 - args.pivot_img_size > 0:
                            #bottom left
                            cv2_im[y1:y1+args.pivot_img_size, x0-args.pivot_img_size:x0, :3] = display_img

                            overlay = cv2_im.copy(); opacity = 0.4
                            cv2.rectangle(cv2_im,(x0-args.pivot_img_size,y1-20),(x0, y1),(46,200,255),cv2.FILLED)
                            cv2.addWeighted(overlay, opacity, cv2_im, 1 - opacity, 0, cv2_im)

                            cv2.putText(cv2_im, label, (x0 - args.pivot_img_size, y1-10), args.font, 0.5, args.text_color, 1)

                            #connect face and text
                            cv2.line(cv2_im,(x0+int(w/2), y1), (x0+int(w/2)-int(w/4), y1+int(args.pivot_img_size/2)),(67,67,67),1)
                            cv2.line(cv2_im, (x0+int(w/2)-int(w/4), y1+int(args.pivot_img_size/2)), (x0, y1+int(args.pivot_img_size/2)), (67,67,67),1)

                        elif y0 - args.pivot_img_size > 0 and x0 - args.pivot_img_size > 0:
                            #top left
                            cv2_im[y0-args.pivot_img_size:y0, x0-args.pivot_img_size:x0, :3] = display_img

                            overlay = cv2_im.copy(); opacity = 0.4
                            cv2.rectangle(cv2_im,(x0 - args.pivot_img_size,y0),(x0, y0+20),(46,200,255),cv2.FILLED)
                            cv2.addWeighted(overlay, opacity, cv2_im, 1 - opacity, 0, cv2_im)

                            cv2.putText(cv2_im, label, (x0 - args.pivot_img_size, y0+10), args.font, 0.5, args.text_color, 1)

                            #connect face and text
                            cv2.line(cv2_im,(x0+int(w/2), y0), (x0+int(w/2)-int(w/4), y0-int(args.pivot_img_size/2)),(67,67,67),1)
                            cv2.line(cv2_im, (x0+int(w/2)-int(w/4), y0-int(args.pivot_img_size/2)), (x0, y0 - int(args.pivot_img_size/2)), (67,67,67),1)

                        elif x1+args.pivot_img_size < resolution_x and y1 + args.pivot_img_size < resolution_y:
                            #bottom right
                            cv2_im[y1:y1+args.pivot_img_size, x1:x1+args.pivot_img_size, :3] = display_img

                            overlay = cv2_im.copy(); opacity = 0.4
                            cv2.rectangle(cv2_im,(x1,y1-20),(x1+args.pivot_img_size, y1),(46,200,255),cv2.FILLED)
                            cv2.addWeighted(overlay, opacity, cv2_im, 1 - opacity, 0, cv2_im)

                            cv2.putText(cv2_im, label, (x1, y1-10), args.font, 0.5, args.text_color, 1)

                            #connect face and text
                            cv2.line(cv2_im,(x0+int(w/2), y1), (x0+int(w/2)+int(w/4), y1+int(args.pivot_img_size/2)),(67,67,67),1)
                            cv2.line(cv2_im, (x0+int(w/2)+int(w/4), y1+int(args.pivot_img_size/2)), (x1, y1+int(args.pivot_img_size/2)), (67,67,67),1)
                    except Exception as err:
                        print(str(err))

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
                        print(id, id2cnt[id].most_common())
                        if id2cnt[id][label] >= args.match_delay:
                            id2identity[id] = (suspect_name if label != 'Unknown' else None, 
                                                label,
                                                best_similarity)
                            del id2cnt[id]
                    except Exception as err:
                        print(str(err))

            # 高乘載管制:1
            break
        
        for id in list(id2identity.keys()): 
            if id not in ids:
                del id2identity[id]

        res = '_'.join(face_names) + "(Press 'q' to quit)"
        if res != prev_res: cv2.destroyAllWindows()
        cv2.imshow(res, cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_res = res

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
