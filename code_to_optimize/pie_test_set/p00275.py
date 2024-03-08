def problem_p00275():
    while 1:

        n = int(eval(input()))

        if n == 0:
            break

        cs = eval(input())

        a, b = [0] * n, 0

        for i in range(100):

            if cs[i] == "M":

                a[i % n] += 1

            elif cs[i] == "S":

                a[i % n], b = 0, b + 1 + a[i % n]

            elif cs[i] == "L":

                a[i % n], b = a[i % n] + 1 + b, 0

        a.sort()

        print((" ".join(map(str, a + [b]))))


problem_p00275()
