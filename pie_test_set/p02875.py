def problem_p02875(input_data):
    n = int(eval(input_data))
    l = [0, 1]
    a = 0
    b = c = 1
    p = 998244353

    for i in range(2, n):

        l += [l[p % i] * (p - int(p / i)) % p]

    for i in range(n, n // 2, -1):

        a += b * c % p
        b = 2 * b % p
        c = c * i * l[n + 1 - i] % p

    return (pow(3, n, p) - 2 * a + p) % p
