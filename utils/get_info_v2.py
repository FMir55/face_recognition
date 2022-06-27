import asyncio

import cv2
import pandas as pd

from utils.apis_v3 import get_face_age, get_face_gender
from utils.bpm import get_bpm
from utils.inference_v3 import inference_embedding, inference_emotion
from utils.similarity_v2 import calc_dist
from utils.thread import get_loop_thread

loop_gender = get_loop_thread()
loop_age = get_loop_thread()

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
        
    '''
    if id in id2info: 
        age, gender = id2info[id].values()
    else:
        try:
            loop = asyncio.get_event_loop()
            files = img2files(img_bgr)
            tasks = [
                loop.create_task(get_face_age(loop, files)),
                loop.create_task(get_face_gender(loop, files))
            ]
            
            age, gender = loop.run_until_complete(
                asyncio.gather(
                    *tasks, 
                    return_exceptions=True
                )
            )
            loop.close()
            id2info[id] = {
                        "age" : age, 
                        "gender" : gender
                    }
        except Exception as err:
            print(str(err))
            age, gender = '', ''

    return (
        f"{age} y",
        f"{gender}({'男' if gender == 'Male' else '女'})"
    )
    '''


"""
Legacy
"""


def get_bpm_emotion(processor, crop_bgr):
    loop = asyncio.get_event_loop()
    tasks = [
        loop.create_task(get_bpm(loop, processor, crop_bgr)),
        loop.create_task(inference_emotion(loop, crop_bgr))
    ]
    
    text_bpm, emotion = loop.run_until_complete(
        asyncio.gather(
            *tasks, 
            return_exceptions=True
        )
    )
    loop.close()
    return text_bpm, emotion


def get_embeddings(suspects):
    embeddings = []
    for suspect in suspects:
        img = cv2.imread(suspect)
        embeddings.append(
            (
                suspect,
                inference_embedding(img)
            )
        )
    df = pd.DataFrame(embeddings, columns = ['suspect', 'embedding_template'])
    return df

def match(df, crop_bgr):
    df['embedding_sample'] = [inference_embedding(crop_bgr)] * len(df)
    df['distance'] = df.apply(calc_dist, axis = 1)
    candidate = df.sort_values(by = ["distance"]).iloc[0]
    suspect_name = candidate['suspect']
    best_distance = candidate['distance']
    best_similarity = int((1 - best_distance)* 100)
    return suspect_name, best_similarity


    
    
