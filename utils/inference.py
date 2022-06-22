import cv2
import numpy as np
import pandas as pd
from pycoral.adapters.common import input_size, output_tensor
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import run_inference


def inference_detection(cv2_im, interpreter_detection, threshold):
    inference_size_detection = input_size(interpreter_detection)
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size_detection)
    run_inference(interpreter_detection, cv2_im_rgb.tobytes())
    return get_objects(interpreter_detection, threshold)

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

def inference_embedding(cv2_im, interpreter_emb):
    inference_size_emb = input_size(interpreter_emb)
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size_emb)
    # aligned_images = prewhiten(cv2_im_rgb[np.newaxis]).astype(np.float32)
    # aligned_images = prewhiten(cv2_im_rgb).astype(np.float32)
    aligned_images = cv2_im_rgb.astype(np.float32)
    run_inference(interpreter_emb, aligned_images.tobytes())
    return output_tensor(interpreter_emb, 0)[0]

def get_embeddings_v2(suspects, interpreter_emb):
    embeddings = []
    for suspect in suspects:
        img = cv2.imread(suspect)
        embeddings.append(
            (
                suspect,
                inference_embedding(img, interpreter_emb)
            )
        )
    df = pd.DataFrame(embeddings, columns = ['suspect', 'embedding_template'])
    return df
