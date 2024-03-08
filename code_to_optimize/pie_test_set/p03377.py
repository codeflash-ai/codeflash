def problem_p03377():
    A, B, X = list(map(int, input().split()))

    print(("YES" if A <= X and X <= A + B else "NO"))


problem_p03377()
