def problem_p03129():
    n, k = list(map(int, input().split()))

    if n == 1 or n == 2:

        if k == 1:

            print("YES")

        else:

            print("NO")

    elif n % 2 == 0:

        cnt = n / 2

        if cnt >= k:

            print("YES")

        else:

            print("NO")

    else:

        cnt = n // 2 + 1

        if cnt >= k:

            print("YES")

        else:

            print("NO")


problem_p03129()
