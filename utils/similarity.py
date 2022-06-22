import re

import numpy as np
from scipy.spatial import distance


def calc_dist(row):
  emb1 = row['embedding_sample']
  emb2 = row['embedding_template']
  print(emb1)
  print(emb2)
  # cosine dist 0~1
  return distance.cosine(emb1, emb2)


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def findDistance(row):
  distance_metric = row['distance_metric']
  img1_representation = row['embedding_sample']
  img2_representation = row['embedding_template']

  distance = 1000 #initialize very large value
  if distance_metric == 'cosine':
    distance = findCosineDistance(img1_representation, img2_representation)
  elif distance_metric == 'euclidean':
    distance = findEuclideanDistance(img1_representation, img2_representation)
  elif distance_metric == 'euclidean_l2':
    distance = findEuclideanDistance(l2_normalize(img1_representation), l2_normalize(img2_representation))

  return distance

def get_label(suspect_name):
  label = suspect_name.split("/")[-1].replace(".jpg", "")
  return re.sub('[0-9]', '', label)
