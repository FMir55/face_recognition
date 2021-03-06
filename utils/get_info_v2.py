import asyncio
from collections import Counter

import cv2
import pandas as pd
import requests

from utils.apis_v3 import get_face_age, get_face_gender
from utils.bpm import get_pulse, run_bpm
from utils.config import Args
from utils.inference_v3 import inference_embedding_prep, inference_emotion
from utils.preparation_v2 import get_suspects
from utils.similarity_v2 import calc_dist, get_label
from utils.thread import get_loop_thread

args = Args()

loop_gender = get_loop_thread()
loop_age = get_loop_thread()
loop_identity = get_loop_thread()
loop_bpm = get_loop_thread()

def get_embeddings():
    suspects = get_suspects()
    embeddings = []
    for suspect in suspects:
        img = cv2.imread(suspect)
        embeddings.append(
            (
                suspect,
                inference_embedding_prep(img)
            )
        )
    df = pd.DataFrame(embeddings, columns = ['suspect', 'embedding_template'])
    return df

# get face embeddings
df = get_embeddings()

def do_identity():
    # At least one template exists
    return df is not None and df.shape[0] > 0

def img2files(img_bgr, fname='sample.jpg'):
    _, encoded_image = cv2.imencode('.jpg', img_bgr)
    return {
        "image": (fname, encoded_image.tobytes())
    }

def get_age_gender(id, id2info, img_bgr):
    if not id in id2info:
        id2info[id] = {key:None for key in ['age', 'gender']}
        files = img2files(img_bgr)
        asyncio.run_coroutine_threadsafe(
            get_face_age(loop_age, files, id2info[id]),
            loop_age
        )
        asyncio.run_coroutine_threadsafe(
            get_face_gender(loop_gender, files, id2info[id]),
            loop_gender
        )

def get_bpm_emotion(id, id2bpm, id2emotion, crop_bgr, idx_emotion):
    if id not in id2bpm:
        id2bpm[id] = get_pulse(args.bpm_limits)
    if id not in id2emotion:
        id2emotion[id] = ''

    asyncio.run_coroutine_threadsafe(
            run_bpm(loop_bpm, id2bpm[id], crop_bgr),
            loop_bpm
        )
    
    if idx_emotion == 0:
        id2emotion[id] = inference_emotion(crop_bgr)

async def match(loop, crop_bgr, id, id2identity):

    emb = inference_embedding_prep(crop_bgr)

    df['embedding_sample'] = [emb] * len(df)
    df['distance'] = df.apply(calc_dist, axis = 1)
    candidate = df.sort_values(by = ["distance"]).iloc[0]
    suspect_name = candidate['suspect']
    best_distance = candidate['distance']
    best_similarity = int((1 - best_distance)* 100)
    label = get_label(suspect_name, best_similarity) if best_similarity >= args.similarity_thresh else f"Guest{id}"
    suspect_name = suspect_name if best_similarity >= args.similarity_thresh else None
    id2identity[id][(suspect_name, label)] += 1

def get_identity(id, id2identity, img_bgr):
    if id not in id2identity:
        id2identity[id] = Counter()
        asyncio.run_coroutine_threadsafe(
                match(loop_identity, img_bgr, id, id2identity),
                loop_identity
            )
    else:
        most = id2identity[id].most_common(1)
        if len(most) != 0 and most[0][1] < args.match_delay:
            asyncio.run_coroutine_threadsafe(
                match(loop_identity, img_bgr, id, id2identity),
                loop_identity
            )


    
    
