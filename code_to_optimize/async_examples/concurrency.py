import asyncio
import time
import random


async def fake_api_call(delay, data):
    await asyncio.sleep(delay)
    return f"Processed: {data}"


async def cpu_bound_task(n):
    result = 0
    for i in range(n):
        result += i ** 2
    return result


async def some_api_call(urls):
    tasks = [
        fake_api_call(random.uniform(0.5, 2.0), url)
        for i, url in enumerate(urls)
    ]
    return await asyncio.gather(*tasks)


async def inefficient_task_creation():
    results = []
    for i in range(10):
        task = asyncio.create_task(fake_api_call(0.5, f"data_{i}"))
        result = await task
        results.append(result)
    
    return results


async def manga():
    results = []
    
    for i in range(5):
        async_result = await fake_api_call(0.3, f"async_{i}")
        results.append(async_result)
        
        time.sleep(0.5)
        cpu_result = sum(range(100000))
        results.append(f"CPU result: {cpu_result}")
    
    return results


if __name__ == "__main__":
    async def main():
        print("Running inefficient concurrency examples...")
        
        urls = [f"https://api.example.com/data/{i}" for i in range(5)]
        start_time = time.time()
        results = await sequential_api_calls(urls)
        print(f"Sequential calls took: {time.time() - start_time:.2f}s")
        
        start_time = time.time()
        results = await inefficient_task_creation()
        print(f"Inefficient tasks took: {time.time() - start_time:.2f}s")
        
    asyncio.run(main())