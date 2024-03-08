def problem_p02732():
    N = int(eval(input()))

    A = list(map(int, input().split()))

    dic = {}

    for a in A:

        if a in dic:

            dic[a] += 1

        else:

            dic[a] = 1

    c = 0

    for i in list(dic.values()):

        c += i * (i - 1) / 2

    for j in range(1, N + 1):

        d = c - dic[A[j - 1]] + 1

        print((int(d)))


problem_p02732()
