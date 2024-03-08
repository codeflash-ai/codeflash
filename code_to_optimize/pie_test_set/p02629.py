def problem_p02629():
    n = int(eval(input()))

    a = []

    while n:

        a += (n % 26,)

        n //= 26

    a = a[::-1]

    for i in range(len(a) - 1, 0, -1):

        if a[i] < 1:

            a[i] += 26

            a[i - 1] -= 1

    print(("".join(chr(c + 96) for c in a[not a[0] :])))


problem_p02629()
