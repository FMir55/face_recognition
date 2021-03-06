import requests


async def get_face_age(loop, files, info,\
    url="https://heartrate.ap-mic.com/get_face_age"):

    response = await loop.run_in_executor(
        None,
        lambda: requests.post(url, files=files)
    )
    age = response.json()['age']
    info['age'] = f"{age} y"

async def get_face_gender(loop, files, info,\
    url="https://heartrate.ap-mic.com/get_face_gender"):

    response = await loop.run_in_executor(
        None,
        lambda: requests.post(url, files=files)
    )
    gender = response.json()['gender']
    info['gender'] = f"{gender}({'男' if gender == 'Male' else '女'})"

