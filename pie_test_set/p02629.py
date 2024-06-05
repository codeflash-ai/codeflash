def problem_p02629(input_data):
    n = int(eval(input_data))

    a = []

    while n:

        a += (n % 26,)

        n //= 26

    a = a[::-1]

    for i in range(len(a) - 1, 0, -1):

        if a[i] < 1:

            a[i] += 26

            a[i - 1] -= 1

    return "".join(chr(c + 96) for c in a[not a[0] :])
