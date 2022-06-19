import cv2
import numpy as np
from deepface import DeepFace
from deepface.commons import distance as dst
from deepface.commons import functions
from deepface.extendedmodels import Age

'''
from deepface.detectors import FaceDetector
'''
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference


class Args:
    model = 'all_models/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    camera_idx = 2
    threshold = 0.1
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def get_models():
    return (
        DeepFace.build_model('Emotion')
        age_model = DeepFace.build_model('Age')
        gender_model = DeepFace.build_model('Gender')
    )

args = Args()
emotion_model, age_model, gender_model = get_models()

def main():
    
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    inference_size = input_size(interpreter)

    cap = cv2.VideoCapture(args.camera_idx)
    cap.set(3, 1920)
    cap.set(4, 1080)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)
        cv2_im = append_objs_to_img(cv2_im, inference_size, objs, args.detector_backend)

        res = "test"
        cv2.imshow(res, cv2_im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def get_face_attr(crop_bgr):
    gray_img = functions.preprocess_face(img = crop_bgr, target_size = (48, 48), grayscale = True, enforce_detection = False, detector_backend = 'ssd')
    face_224 = functions.preprocess_face(img = crop_bgr, target_size = (224, 224), grayscale = False, enforce_detection = False, detector_backend = 'ssd')

    # emotion
    emotion_predictions = emotion_model.predict(gray_img)[0,:]
    emotion = args.emotion_labels[np.argmax(emotion_predictions)]

    # age
    age_predictions = age_model.predict(face_224)[0,:]
    apparent_age = Age.findApparentAge(age_predictions)
    age = int(round(apparent_age))

    # gender
    gender_prediction = gender_model.predict(face_224)[0,:]
    idx = np.argmax(gender_prediction)
    gender = "Female" if idx == 0 else "Male"

    return emotion, age, gender

def append_objs_to_img(cv2_im, inference_size, objs):
    cv2_im_clean = cv2_im.copy()
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, 'face')

        crop_bgr = cv2_im_clean[y0:y1, x0:x1]
        emotion, age, gender = get_face_attr(crop_bgr)

        # draw
        attr = f"{gender}, {age}y, {emotion}"
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        # cv2.putText(bbox_array, text, (left+5, top-10), 0, 0.6, color_text, thickness=2)
        cv2.putText(cv2_im, attr, (x+5, y-10), 0, 0.6, (0, 0, 255), thickness=2)
        cv2_im = cv2.putText(cv2_im, attr, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()
