import cv2
from pycoral.adapters.common import input_size, output_tensor
from pycoral.utils.edgetpu import make_interpreter, run_inference


class Args:
    model = '../all_models/facenet.tflite'

args = Args()

def main():
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    # 160 160
    inference_size = input_size(interpreter)

    '''
    cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
    cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
    '''
    import numpy as np
    cv2_im_rgb = np.ones((160, 160, 3))
    run_inference(interpreter, cv2_im_rgb.tobytes())
    embedding = output_tensor(interpreter, -1)
    print(embedding)


if __name__ == '__main__':
    main()
