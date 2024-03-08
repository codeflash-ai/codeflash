def problem_p00516():
    n, k = list(map(int, input().split()))

    a = [eval(input()) for i in range(n + k)]

    cnt = {i: 0 for i in range(n)}

    for i in range(n, n + k):

        for j in range(n):

            if a[i] >= a[j]:

                cnt[j] += 1

                break

    print(sorted(list(cnt.items()), key=lambda x: x[1], reverse=True)[0][0] + 1)


problem_p00516()
