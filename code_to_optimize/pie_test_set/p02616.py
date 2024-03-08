def problem_p02616():
    mod = 10**9 + 7

    n, k = list(map(int, input().split()))

    arr = list(map(int, input().split()))

    arr = sorted(arr, reverse=True, key=lambda x: abs(x))

    if k == n:

        ans = 1

        for i in range(n):

            ans *= arr[i]

            ans %= mod

        print(ans)

    else:

        if k % 2 == 1 and max(arr) < 0:

            ans = 1

            for i in range(k):

                ans *= arr[n - 1 - i]

                ans %= mod

            print(ans)

        else:

            ans = 1

            cnt = 0

            for i in range(k):

                if arr[i] < 0:

                    cnt += 1

                ans *= arr[i]

                ans %= mod

            if cnt % 2 == 0:

                print(ans)

            else:

                min_plus = -1

                min_minus = 1

                for i in range(k):

                    if arr[i] >= 0:

                        min_plus = arr[i]

                    else:

                        min_minus = arr[i]

                max_plus = -1

                max_minus = 1

                for i in range(k, n):

                    if arr[i] >= 0 and max_plus == -1:

                        max_plus = arr[i]

                    if arr[i] < 0 and max_minus == 1:

                        max_minus = arr[i]

                if min_plus == -1:

                    arr.remove(min_minus)

                    ans = 1

                    for i in range(k - 1):

                        ans *= arr[i]

                        ans %= mod

                    ans *= max_plus

                    ans %= mod

                elif min_minus == 1:

                    arr.remove(min_plus)

                    ans = 1

                    for i in range(k - 1):

                        ans *= arr[i]

                        ans %= mod

                    ans *= max_minus

                    ans %= mod

                elif min_plus * max_plus >= min_minus * max_minus:

                    arr.remove(min_minus)

                    ans = 1

                    for i in range(k - 1):

                        ans *= arr[i]

                        ans %= mod

                    ans *= max_plus

                    ans %= mod

                else:

                    arr.remove(min_plus)

                    ans = 1

                    for i in range(k - 1):

                        ans *= arr[i]

                        ans %= mod

                    ans *= max_minus

                    ans %= mod

                print(ans)


problem_p02616()
