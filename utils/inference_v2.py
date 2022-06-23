import cv2
import numpy as np
import pandas as pd
from pycoral.adapters.classify import get_classes
from pycoral.adapters.common import input_size, output_tensor
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference

from utils.apis import get_face_age, get_face_info
from utils.config import Args
from utils.preprocess import preprocess_244, preprocess_gray


def get_interpreter(path):
  interpreter = make_interpreter(path)
  interpreter.allocate_tensors()
  return interpreter

args = Args()

# face detection
interpreter_detection = get_interpreter(args.model_detection)
inference_size_detection = input_size(interpreter_detection)

# face embedding
interpreter_emb = get_interpreter(args.model_emb)

# face attribute
interpreter_gender = get_interpreter(args.model_gender)
# interpreter_age = get_interpreter(args.model_age)
interpreter_emotion = get_interpreter(args.model_emotion)


def inference_emotion(
    cv2_im, 
    labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    ):
    inference_size_emotion = input_size(interpreter_emotion)
    crop_gray = preprocess_gray(cv2_im, inference_size_emotion)
    run_inference(interpreter_emotion, crop_gray.tobytes())
    c = get_classes(interpreter_emotion, top_k=1)[0]
    return labels[c.id]

def inference_gender(
    crop_224,
    labels = ['Female', 'Male']
    ):
    run_inference(interpreter_gender, crop_224.tobytes())
    c = get_classes(interpreter_gender, top_k=1)[0]
    return labels[c.id]

def get_attr_v3(id, id2info, crop_bgr):
    # emotion
    emotion = inference_emotion(crop_bgr)

    # age/gender
    if id in id2info: 
        age, gender = id2info[id].values()
    else:
        try:
            face_info = get_face_info(crop_bgr)
            _, age, gender = face_info.values()
            
            id2info[id] = {
                "age" : age, 
                "gender" : gender
            }
        except Exception as err:
            print(str(err))
            age, gender = '', ''

    return f"{gender}, {age}y, {emotion}"

def get_attr_v2(id, id2info, crop_bgr):
    # emotion
    emotion = inference_emotion(crop_bgr)

    # age/gender
    if id in id2info: 
        age, gender = id2info[id].values()
    else:
        try:
            # gender
            crop_224 = preprocess_244(crop_bgr)
            gender = inference_gender(crop_224)

            # age
            age = get_face_age(crop_bgr)
            
            id2info[id] = {
                "age" : age, 
                "gender" : gender
            }
        except Exception as err:
            print(str(err))
            age, gender = '', ''

    return f"{gender}, {age}y, {emotion}"

'''
def inference_age(crop_224):
    run_inference(interpreter_age, crop_224.tobytes())
    age_predictions = output_tensor(interpreter_age, 0)[0].copy()
    apparent_age = findApparentAge(age_predictions)
    return int(round(apparent_age))

def get_attr(id, id2info, crop_bgr):
    # emotion
    emotion = inference_emotion(crop_bgr)

    # age/gender
    if id in id2info: 
        age, gender = id2info[id].values()
    else:
        crop_224 = preprocess_244(crop_bgr)
        gender = inference_gender(crop_224)
        age = inference_age(crop_224)
        id2info[id] = {
            "age" : age, 
            "gender" : gender
        }

    return f"{gender}, {age}y, {emotion}"
'''

def inference_detection(cv2_im, threshold):
    inference_size_detection = input_size(interpreter_detection)
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size_detection)
    run_inference(interpreter_detection, cv2_im_rgb.tobytes())
    objs = get_objects(interpreter_detection, threshold)

    height, width, _ = cv2_im.shape
    scale_x, scale_y = width / inference_size_detection[0], height / inference_size_detection[1]
    return objs, scale_x, scale_y

def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))
    y = (x - mean) / std_adj
    return y

def inference_embedding(cv2_im):
    inference_size_emb = input_size(interpreter_emb)
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size_emb)
    aligned_images = prewhiten(cv2_im_rgb[np.newaxis]).astype(np.float32)
    run_inference(interpreter_emb, aligned_images.tobytes())
    return output_tensor(interpreter_emb, 0)[0].copy()

def get_embeddings_v2(suspects):
    embeddings = []
    for suspect in suspects:
        img = cv2.imread(suspect)
        embeddings.append(
            (
                suspect,
                inference_embedding(img)
            )
        )
    df = pd.DataFrame(embeddings, columns = ['suspect', 'embedding_template'])
    return df
