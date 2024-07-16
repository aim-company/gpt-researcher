import concurrent
import asyncio


async def run_in_thread(func):
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await asyncio.get_running_loop().run_in_executor(pool, func)
