def problem_p00011():
    w = eval(input())

    list = list(range(0, w + 1))

    n = eval(input())

    for i in range(n):

        a, b = list(map(int, input().split(",")))

        tmp = list[a]

        list[a] = list[b]

        list[b] = tmp

    for i in range(w):

        print(list[i + 1])


problem_p00011()
