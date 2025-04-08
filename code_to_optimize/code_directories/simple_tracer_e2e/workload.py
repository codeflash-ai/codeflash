from concurrent.futures import ThreadPoolExecutor


def funcA(number):
    k = 0
    for i in range(number * 100):
        k += i
    # Simplify the for loop by using sum with a range object
    j = sum(range(number))

    # Use a generator expression directly in join for more efficiency
    return " ".join(str(i) for i in range(number))


def test_threadpool() -> None:
    pool = ThreadPoolExecutor(max_workers=3)
    args = list(range(10, 31, 10))
    result = pool.map(funcA, args)

    for r in result:
        print(r)


if __name__ == "__main__":
    test_threadpool()
