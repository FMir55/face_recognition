import cv2
import numpy as np
from pycoral.adapters.classify import get_classes
from pycoral.adapters.common import input_size, output_tensor
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import make_interpreter, run_inference

from utils.config import Args
from utils.preprocess import preprocess_gray


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

async def inference_embedding(loop, cv2_im, emotion):
    inference_size_emb = input_size(interpreter_emb)
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size_emb)
    aligned_images = prewhiten(cv2_im_rgb[np.newaxis]).astype(np.float32)
    await loop.run_in_executor(
        None,
        run_inference, 
        interpreter_emb, aligned_images.tobytes()
    )
    emotion = output_tensor(interpreter_emb, 0)[0].copy()

def inference_embedding_prep(cv2_im):
    inference_size_emb = input_size(interpreter_emb)
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size_emb)
    aligned_images = prewhiten(cv2_im_rgb[np.newaxis]).astype(np.float32)
    run_inference(interpreter_emb, aligned_images.tobytes())
    await loop.run_in_executor(
        None,
        run_inference, 
        interpreter_emb, aligned_images.tobytes()
    )
    return output_tensor(interpreter_emb, 0)[0].copy()
'''
async def get_face_age(loop, files, info,\
    url="https://heartrate.ap-mic.com/get_face_age"):

    # response = await requests.post(url, files=files)
    response = await loop.run_in_executor(
        None,
        lambda: requests.post(url, files=files)
    )
    age = response.json()['age']
    info['age'] = f"{age} y"
'''


