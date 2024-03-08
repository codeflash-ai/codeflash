def problem_p03036():
    r, d, x = list(map(int, input().split()))

    for i in range(10):

        temp = r * x - d

        print(temp)

        x = temp


problem_p03036()
