import requests


async def get_face_age(loop, files, \
    url="https://heartrate.ap-mic.com/get_face_age"):

    response = await loop.run_in_executor(
        None,
        lambda: requests.post(url, files=files)
    )
    return response.json()['age']

async def get_face_gender(loop, files, \
    url="https://heartrate.ap-mic.com/get_face_gender"):

    response = await loop.run_in_executor(
        None,
        lambda: requests.post(url, files=files)
    )
    return response.json()['gender']

