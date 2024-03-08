def problem_p02835():
    a, b, c = list(map(int, input().split()))

    print(("bust" if a + b + c >= 22 else "win"))


problem_p02835()
