import asyncio
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
