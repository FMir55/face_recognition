import cv2
import numpy as np
import pandas as pd
import requests


def get_face_info(img_bgr, fname='sample.jpg'):
  _, encoded_image = cv2.imencode('.jpg', img_bgr)

  files = {
      "image": (fname, encoded_image.tobytes())
  }

  response = requests.post(
      "https://heartrate.ap-mic.com/get_face_info",
      files=files
  )
  
  return response.json()

def get_embedding(img, fname = 'sample.jpg'):
    _, encoded_image = cv2.imencode('.jpg', img)

    files = {
        "image": (fname, encoded_image.tobytes())
    }

    response = requests.post(
        "https://heartrate.ap-mic.com/get_face_embedding",
        files=files
    )
    
    return np.array(response.json()['embedding'])

def get_embeddings(suspects, distance_metric="cosine"):
  embeddings = []
  for suspect in suspects:
    img = cv2.imread(suspect)
    embeddings.append(
        (
            suspect,
            get_embedding(img)
        )
    )
    df = pd.DataFrame(embeddings, columns = ['suspect', 'embedding_template'])
    df['distance_metric'] = distance_metric
    return df

