def problem_p01931():
    # AOJ 2831: Check answers

    # Python3 2018.7.12 bal4u

    n = int(eval(input()))

    if n == 0:
        print((0))

    else:

        s = eval(input())

        ans = 1

        for i in range(1, len(s)):

            if s[i - 1] == "x" and s[i] == "x":
                break

            ans += 1

        print(ans)


problem_p01931()
