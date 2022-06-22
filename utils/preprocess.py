# from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image


def preprocess_gray(im, target_size=(48, 48)):
  img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

  factor_0 = target_size[0] / img.shape[0]
  factor_1 = target_size[1] / img.shape[1]
  factor = min(factor_0, factor_1)

  dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
  img = cv2.resize(img, dsize)

  # Then pad the other side to the target size by adding black pixels
  diff_0 = target_size[0] - img.shape[0]
  diff_1 = target_size[1] - img.shape[1]

  img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

  #double check: if target image is not still the same size with target.
  if img.shape[0:2] != target_size:
    img = cv2.resize(img, target_size)

  img_pixels = image.img_to_array(img) #what this line doing? must?
  img_pixels = np.expand_dims(img_pixels, axis = 0)
  img_pixels /= 255 #normalize input in [0, 1]

  return img_pixels

def preprocess_244(img, target_size=(224, 224)):
  factor_0 = target_size[0] / img.shape[0]
  factor_1 = target_size[1] / img.shape[1]
  factor = min(factor_0, factor_1)

  dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
  img = cv2.resize(img, dsize)

  # Then pad the other side to the target size by adding black pixels
  diff_0 = target_size[0] - img.shape[0]
  diff_1 = target_size[1] - img.shape[1]

  # Put the base image in the middle of the padded image
  img = np.pad(img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')

  #double check: if target image is not still the same size with target.
  if img.shape[0:2] != target_size:
    img = cv2.resize(img, target_size)

  img_pixels = image.img_to_array(img) #what this line doing? must?
  img_pixels = np.expand_dims(img_pixels, axis = 0)
  img_pixels /= 255 #normalize input in [0, 1]

  return img_pixels

def findApparentAge(age_predictions):
	output_indexes = np.array([i for i in range(0, 101)])
	apparent_age = np.sum(age_predictions * output_indexes)
	return apparent_age
