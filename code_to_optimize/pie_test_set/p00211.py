def problem_p00211():
    def gcd(a, b):

        return a if b == 0 else gcd(b, a % b)

    while 1:

        n = eval(input())

        if n == 0:
            break

        s = [list(map(int, input().split())) for i in range(n)]

        t = c = 1

        for i in s:
            t = t * i[1] / gcd(t, i[1])

        for i in range(n):

            s[i] = t * s[i][0] / s[i][1]

            c = c * s[i] / gcd(c, s[i])

        for i in s:
            print(c / i)


problem_p00211()
