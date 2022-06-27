import asyncio
import time
from threading import Thread


def get_loop_thread():
    new_loop = asyncio.new_event_loop()
    t = Thread(target=start_loop, args=(new_loop,))
    t.setDaemon(True)
    t.start()
    return new_loop

def start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def get_info(t, info):
    await asyncio.sleep(t)
    info[t] = str(t)
    print(info)

def main():
    loop1 = get_loop_thread()
    loop2 = get_loop_thread()
    for i in range(15):
        if i in [0, 9]:
            info = {}
            asyncio.run_coroutine_threadsafe(get_info(5, info), loop1)
            asyncio.run_coroutine_threadsafe(get_info(3, info), loop2)

        time.sleep(1)
        print(i)
    
    return 0


if __name__ == '__main__':
    main()
