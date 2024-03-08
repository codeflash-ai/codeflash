def problem_p02714():
    import sys

    read = sys.stdin.buffer.read

    readline = sys.stdin.buffer.readline

    readlines = sys.stdin.buffer.readlines

    sys.setrecursionlimit(10**7)

    n = int(readline())

    s = eval(input())

    rgb = [0, 0, 0]

    for i in range(n):

        if s[i] == "R":

            rgb[0] += 1

        elif s[i] == "G":

            rgb[1] += 1

        else:

            rgb[2] += 1

    ans = rgb[0] * rgb[1] * rgb[2]

    for i in range(n):

        for j in range(i + 1, n):

            k = 2 * j - i

            if k >= n:

                break

            if s[i] != s[j] and s[j] != s[k] and s[i] != s[k]:

                ans -= 1

    print(ans)


problem_p02714()
