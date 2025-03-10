from concurrent.futures import ThreadPoolExecutor


def add_numbers(a: int, b: int) -> int:
    print(f"[ADD_NUMBERS] Starting with parameters: a={a}, b={b}")
    result = a + b
    print(f"[ADD_NUMBERS] Returning result: {result}")
    return result


def test_threadpool() -> None:
    print("[TEST_THREADPOOL] Starting thread pool execution")
    pool = ThreadPoolExecutor(max_workers=3)
    numbers = [(10, 20), (30, 40), (50, 60)]
    print("[TEST_THREADPOOL] Submitting tasks to thread pool")
    result = pool.map(add_numbers, *zip(*numbers))

    print("[TEST_THREADPOOL] Processing results")
    for r in result:
        print(f"[TEST_THREADPOOL] Thread result: {r}")
    print("[TEST_THREADPOOL] Finished thread pool execution")


def multiply_numbers(a: int, b: int) -> int:
    print(f"[MULTIPLY_NUMBERS] Starting with parameters: a={a}, b={b}")
    result = a * b
    print(f"[MULTIPLY_NUMBERS] Returning result: {result}")
    return result


if __name__ == "__main__":
    print("[MAIN] Starting testbench execution")

    print("[MAIN] Calling test_threadpool()")
    test_threadpool()
    print("[MAIN] Finished test_threadpool()")

    print("[MAIN] Calling add_numbers(5, 10)")
    result1 = add_numbers(5, 10)
    print(f"[MAIN] add_numbers result: {result1}")

    print("[MAIN] Calling add_numbers(15, 25)")
    result2 = add_numbers(15, 25)
    print(f"[MAIN] add_numbers result: {result2}")

    print("[MAIN] Calling multiply_numbers(3, 7)")
    result3 = multiply_numbers(3, 7)
    print(f"[MAIN] multiply_numbers result: {result3}")

    print("[MAIN] Calling multiply_numbers(5, 9)")
    result4 = multiply_numbers(5, 9)
    print(f"[MAIN] multiply_numbers result: {result4}")

    print("[MAIN] Testbench execution completed")
