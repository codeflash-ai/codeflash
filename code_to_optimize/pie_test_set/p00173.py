def problem_p00173():
    for i in range(9):

        a, b, c = list(map(str, input().split()))

        print(a, int(b) + int(c), 200 * int(b) + 300 * int(c))


problem_p00173()
