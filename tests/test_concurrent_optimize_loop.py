from codeflash.api.aiservice import AiServiceClient
from collections import deque

import concurrent.futures
import time

def a():
    """Function that makes a request and returns list b"""
    # Simulate network request
    time.sleep(3)
    return ["result1", "result2", "result3"]

def process_item(item):
    """Process an individual item from list c"""
    # Simulate processing
    time.sleep(0.5)
    return f"processed_{item}"

def test_main():
    # List c that needs sequential processing
    c = deque(["item1", "item2", "item3", "item4", "item5"])
    done = False
    # Start function a asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(a)
        # Process list c sequentially while occasionally checking if a has completed
        processed_results = []
        iter = 0
        while (c) or (not done):
            print(c,done,iter)
            done = True if future is None else future.done()
            if done and (future is not None):
                b = future.result()
                c.extend(b)
                print(f"Added result from a() to list c: {b}")
                future = None
            if c:
                item = c.popleft()
                processed_results.append(process_item(item))
            iter+=1

    print(f"Final list c: {c}")
    print(f"Processed results: {processed_results}")

def test_concurrent_optimize_loop():
    aiservice1 = AiServiceClient()
    aiservice2 = AiServiceClient()
    opt_candidates1 = deque(aiservice1.optimize_python_code())
    opt_candidates2 = aiservice2.optimize_python_code()
    while opt_candidates1 and (not futures.done()):
        # if response is complete then extend the queue on the right, always pop from the left
        # if opt_candidate2.is_complete:
        #     opt_candidates1.extend(opt_candidates2)
        if futures.done():
            opt_candidates1.extend(future.result())
            futures=None
        opt_candidate1 = opt_candidates1.popleft()
        # do whatever you want with opt candidates


