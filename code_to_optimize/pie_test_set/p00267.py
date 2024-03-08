def problem_p00267():
    # AOJ 0272: The Lonely Girl's Lie

    # Python3 2018.6.26 bal4u

    while True:

        n = int(eval(input()))

        if n == 0:
            break

        a = list(map(int, input().split()))

        b = list(map(int, input().split()))

        a.sort(reverse=True)

        b.sort(reverse=True)

        ans, i = n, -1

        for k in range(0, n, 2):

            i += 1

            if a[k] > b[i]:

                ans = k + 1

                break

        print(("NA" if ans == n else ans))


problem_p00267()
