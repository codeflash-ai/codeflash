def problem_p03011():
    pqr = list(map(int, input().split()))

    pqr.sort()

    print((sum(pqr[:2])))


problem_p03011()
