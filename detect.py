import cv2
import numpy as np
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference

# from utils.apis import get_embedding, get_embeddings, get_face_info
from utils.apis import get_embedding, get_face_info
from utils.similarity import findDistance
from utils.tracker import convert_detection, get_tracker


class Args:
    # camera device index
    camera_idx = 2

    # face detection
    model = 'all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    threshold = 0.1

    # tracker
    max_distance_between_points = 100 #@param {type:"slider", min:1, max:100, step:1}
    initialization_delay = 10

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

    '''
    # get face embeddings
    df = get_embeddings(suspects)
    '''

    id2info = {}
    id2identity = {}
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
            id = tracked_object.id
            x0, y0, x1, y1 = tracked_object.last_detection.points.flatten()
            crop_bgr = cv2_im[y0:y1, x0:x1]
            if id in id2info: 
                emotion, age, gender = id2info[id].values()

            else:
                face_info = get_face_info(crop_bgr)
                emotion, age, gender = face_info.values()
                
                id2info[id] = {
                    "emotion" : emotion, 
                    "age" : age, 
                    "gender" : gender
                }

            '''
            if id in id2identity:
                suspect_name, best_simlarity = id2identity[id]
            else:
                df['embedding_sample'] = [get_embedding(crop_bgr)] * len(df)
                df['distance'] = df.apply(findDistance, axis = 1)
                candidate = df.sort_values(by = ["distance"]).iloc[0]
                suspect_name = candidate['suspect']
                best_distance = candidate['distance']
                best_simlarity = int((1 - best_distance)* 100)

                if identity: 
                    id2identity[id] = (suspect_name, best_simlarity)
            '''
            
            # draw
            attr = f"{gender}, {age}y, {emotion}"
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, attr, (x0, y0+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
 
        # cv2_im = append_objs_to_img(cv2_im, inference_size, objs)

        res = "test"
        cv2.imshow(res, cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

"""
def append_objs_to_img(cv2_im, inference_size, objs):
    cv2_im_clean = cv2_im.copy()
    height, width, _ = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        '''
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, 'face')
        '''

        crop_bgr = cv2_im_clean[y0:y1, x0:x1]

        face_info = get_face_info(crop_bgr)
        emotion, age, gender = face_info.values()

        # draw
        attr = f"{gender}, {age}y, {emotion}"
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, attr, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im
"""

if __name__ == '__main__':
    main()
