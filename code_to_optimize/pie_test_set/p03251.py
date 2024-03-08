def problem_p03251():
    n, m, x, y = list(map(int, input().split()))

    x1 = max([int(i) for i in input().split()]) + 1

    y1 = min([int(i) for i in input().split()])

    if x1 <= y1 and x1 <= y and y1 > x:

        print("No War")

    else:

        print("War")


problem_p03251()
