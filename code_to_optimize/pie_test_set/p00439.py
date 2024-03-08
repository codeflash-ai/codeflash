def problem_p00439():
    while True:

        n, k = list(map(int, input().split()))

        if n == 0 and k == 0:

            break

        lst = []

        for i in range(n):

            lst.append(int(eval(input())))

        count = 0

        for i in range(k):

            count += lst[i]

        ans = count

        for i in range(k, n):

            count += lst[i]

            count -= lst[i - k]

            ans = max(ans, count)

        print(ans)


problem_p00439()
