def problem_p03733():
    N, T = list(map(int, input().split()))

    t = list(map(int, input().split()))

    cnt = 0

    for i in range(1, N):

        emp_time = (t[i] - t[i - 1]) - T

        if emp_time >= 0:

            cnt += T

        else:

            cnt += t[i] - t[i - 1]

    print((cnt + T))


problem_p03733()
