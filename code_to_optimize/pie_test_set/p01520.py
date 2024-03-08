def problem_p01520():
    n, t, e = list(map(int, input().split()))

    x = list(map(int, input().split()))

    for i in range(n):

        if t % x[i] <= e or x[i] - t % x[i] <= e:

            print(i + 1)

            break

    else:

        print(-1)


problem_p01520()
