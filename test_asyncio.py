import asyncio
import time

import aiohttp
import cv2
import requests

'''
def send_req(url = 'https://www.google.com.tw/'):
    return requests.get(url)

async def send_req2(loop, i, url = 'https://www.google.com.tw/'):
    return (
        await loop.run_in_executor(None,requests.get,url),
        i
    )
'''

def get_detections(crop_bgr, \
    url = 'https://heartrate.ap-mic.com/get_face_embedding', fname='sample.jpg'):
    
    _, encoded_image = cv2.imencode('.jpg', crop_bgr)
    files = {
        "image": (fname, encoded_image.tobytes())
    }
    return requests.post(url, files=files)

async def get_detections2(loop, crop_bgr, \
    url = 'https://heartrate.ap-mic.com/get_face_embedding', fname='sample.jpg'):
    
    _, encoded_image = cv2.imencode('.jpg', crop_bgr)
    files = {
        "image": (fname, encoded_image.tobytes())
    }

    return await loop.run_in_executor(
        None,
        lambda: requests.post(url, files=files)
    )

def main():
    crop_bgr = cv2.imread('../test_samples/face.png')
    # detection 1
    start_time = time.time()
    for _ in range(5):
        res = get_detections(crop_bgr)
        print(res)
    print(time.time() - start_time)

    # detection 2
    loop = asyncio.get_event_loop()
    start_time = time.time() 
    tasks = [loop.create_task(get_detections2(loop, crop_bgr)) for _ in range(5)]

    results = loop.run_until_complete(
        asyncio.gather(
            *tasks, 
            return_exceptions=True
        )
    ) 
    for result in results: print(result) 
    print(time.time() - start_time)


    """
    # sync
    start_time = time.time()
    for _ in range(10):
        res = send_req()
        print(res)
    print(time.time() - start_time)

    # async wait
    loop = asyncio.get_event_loop()
    start_time = time.time()
    tasks = [loop.create_task(send_req2(loop, i)) for i in range(10)]

    finished, unfinished = loop.run_until_complete(
        asyncio.wait(
            tasks, 
            return_when=asyncio.FIRST_COMPLETED
            )
        )
    for task in finished: print(task.result())
    print("unfinished:", len(unfinished))

    finished, unfinished2 = loop.run_until_complete(
        asyncio.wait(
            unfinished
            )
        )
    for task in finished: print(task.result())
    print("unfinished2:", len(unfinished2))

    print(time.time() - start_time)
    # loop.close()

    # async gather
    loop = asyncio.get_event_loop()
    start_time = time.time()
    tasks = [loop.create_task(send_req2(loop, i)) for i in range(10)]

    results = loop.run_until_complete(
        asyncio.gather(
            *tasks, 
            return_exceptions=True
        )
    ) 
    for result in results: print(result) 
    print(time.time() - start_time)
    # loop.close()
    """


if __name__ == '__main__':
    main()
