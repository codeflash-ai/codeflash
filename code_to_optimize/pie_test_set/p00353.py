def problem_p00353():
    m, f, b = list(map(int, input().split()))

    print(("NA" if m + f < b else max(0, b - m)))


problem_p00353()
