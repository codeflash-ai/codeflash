def problem_p03403():
    n = int(eval(input()))

    tmpa = list(map(int, input().split()))

    A = [0] * (n + 2)

    A[0] = 0

    A[n + 1] = 0

    for i in range(1, n + 1, 1):
        A[i] = tmpa[i - 1]

    # 普通に総和を出す

    sum = 0

    for i, a in enumerate(A):

        if i == n + 1:
            break  # n+1はゴールなので先に行かない

        sum += abs(a - A[i + 1])  # a:A[i]

    # 引いて足す

    for i in range(1, n + 1, 1):  # 1～Nを省いたとき

        nowsum = sum

        nowsum -= abs(A[i - 1] - A[i])

        nowsum -= abs(A[i] - A[i + 1])

        nowsum += abs(A[i - 1] - A[i + 1])

        print(nowsum)


problem_p03403()
