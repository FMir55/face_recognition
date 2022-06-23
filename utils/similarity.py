import re

import numpy as np
from scipy.spatial import distance


def calc_dist(row):
  emb1 = row['embedding_sample']
  emb2 = row['embedding_template']
  # cosine dist 0~1
  return distance.cosine(emb1, emb2)

def get_label(suspect_name):
  label = suspect_name.split("/")[-1].replace(".jpg", "")
  return re.sub('[0-9]', '', label)
