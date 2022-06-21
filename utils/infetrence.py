import cv2
from pycoral.adapters.detect import get_objects
from pycoral.utils.edgetpu import run_inference


def inference_detection(cv2_im, interpreter_detection, inference_size_detection):
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size_detection)
    run_inference(interpreter_detection, cv2_im_rgb.tobytes())
    return get_objects(interpreter_detection, args.threshold)

def inference_embedding():
    run_inference(interpreter, cv2_im_rgb.tobytes())
    return output_tensor(interpreter, -1)[0]

def inference_embeddings():
    run_inference(interpreter, cv2_im_rgb.tobytes())
    return output_tensor(interpreter, -1)[0]
