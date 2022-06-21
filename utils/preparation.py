import os
import shutil
from pathlib import Path


def mkdir(path):
  if not path.is_dir():
    path.mkdir()

def get_legal_fname(path_identity, identity):
  for idx in range(len(os.listdir(str(path_identity)))+1):
    fname = path_identity / Path(identity + f"{str(idx)}.jpg")
    if not fname.exists():
      return fname

def get_suspects(path):
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
