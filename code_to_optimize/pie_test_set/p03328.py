def problem_p03328():
    a, b = (int(i) for i in input().split())

    print((sum(range(b - a)) - a))


problem_p03328()
