def funcA(number):
    k = 0
    for i in range(number * 100):
        k += i
    # Simplify the for loop by using sum with a range object
    j = sum(range(number))

    # Use a generator expression directly in join for more efficiency
    return " ".join(str(i) for i in range(number))


if __name__ == "__main__":
    for i in range(10, 31, 10):
        funcA(10)
