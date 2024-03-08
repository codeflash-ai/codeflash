def problem_p04047():
    N = int(eval(input()))

    L = list(map(int, input().split(" ")))

    L = sorted(L)

    tmp = 0

    for i in range(N):

        tmp += L[i * 2]

    print(tmp)


problem_p04047()
