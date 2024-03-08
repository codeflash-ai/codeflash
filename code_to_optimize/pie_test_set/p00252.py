def problem_p00252():
    a, b, c = list(map(int, input().split()))

    print((["Close", "Open"][(a & b) ^ c]))


problem_p00252()
