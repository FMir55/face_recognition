import os
from pathlib import Path

import numpy as np

from utils.config import Args

args = Args()

def mkdir(path):
  if not path.is_dir():
    path.mkdir()

def get_legal_fname(path_identity, identity):
  for idx in range(len(os.listdir(str(path_identity)))+1):
    fname = path_identity / Path(identity + f"{str(idx)}.jpg")
    if not fname.exists():
      return fname

def get_suspects(path=args.path_face_db):
  suspects = []
  #check passed db folder exists
  if path.is_dir():
    for r, d, f in os.walk(str(path)): # r=root, d=directories, f = files
      for file in f:
        if ('.jpg' in file):
          #exact_path = os.path.join(r, file)
          exact_path = r + "/" + file
          #print(exact_path)
          suspects.append(exact_path)
  if len(suspects) == 0:
    print("WARNING: There is no image in this path ( ", str(path),") . Face recognition will not be performed.")
  return suspects

def prune(x0, y0, x1, y1):
  x_mid = np.mean([x0, x1])
  y_mid = np.mean([y0, y1])
  h = y1 - y0
  w = h * 0.6
  h *= 0.8

  return (
    max(int(x_mid - w/2), x0),
    max(int(y_mid - h/2), y0),
    min(int(x_mid + w/2), x1),
    min(int(y_mid + h/2), y1)
  )

def clean_dict(cnt, ids):
  ids2remove = set(cnt.keys()) - set(ids)
  for id in ids2remove: del cnt[id]
