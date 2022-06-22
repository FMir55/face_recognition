import cv2
import numpy as np
from pycoral.adapters.common import input_size, output_tensor
from pycoral.utils.edgetpu import make_interpreter, run_inference
from scipy.spatial import distance


class Args:
    model = '../all_models/facenet.tflite'

args = Args()

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

def main():
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    # 160 160
    inference_size = input_size(interpreter)

    cv2_im = cv2.imread('../face_db/a/a0.jpg')
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
    aligned_images =  prewhiten(cv2_im_rgb[np.newaxis]).astype(np.float32)
    run_inference(interpreter, aligned_images.tobytes())
    embedding1 = output_tensor(interpreter, 0)[0].copy()
    print(embedding1)

    cv2_im = cv2.imread('../face_db/a/a1.jpg')
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
    aligned_images =  prewhiten(cv2_im_rgb[np.newaxis]).astype(np.float32)
    run_inference(interpreter, aligned_images.tobytes())
    embedding2 = output_tensor(interpreter, 0)[0].copy()
    print(embedding2)
    
    print(distance.cosine(embedding1, embedding2))

    print(embedding1)
    print(embedding2)


if __name__ == '__main__':
    main()
