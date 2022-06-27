#!/usr/bin/env python3
# asyncq.py

import asyncio
import random
import time


async def randsleep(caller=None) -> None:
    i = random.randint(0, 10)
    if caller:
        print(f"{caller} sleeping for {i} seconds.")
    await asyncio.sleep(i)

async def produce(name: int, q: asyncio.Queue) -> None:
    await asyncio.sleep(2)
    await q.put(name)
    print(f"Producer {name} added to queue.")

async def consume(name: int, q: asyncio.Queue) -> None:
    idx = 0
    while True:
        await asyncio.sleep(1)
        idx += 1
        print(idx)
        i = await q.get()
        print(i)
        q.task_done()

async def main():
    q = asyncio.Queue()
    consumers = [asyncio.create_task(consume(0, q))]
    producers = [
        asyncio.create_task(produce(0, q)),
        asyncio.create_task(produce(1, q))
    ]
    await asyncio.gather(*producers)
    await q.join()  # Implicitly awaits consumers, too
    for c in consumers:
        c.cancel()

if __name__ == "__main__":
    random.seed(444)
    start = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - start
    print(f"Program completed in {elapsed:0.5f} seconds.")
