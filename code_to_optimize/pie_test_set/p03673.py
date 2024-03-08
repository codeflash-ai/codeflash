def problem_p03673():
    N = int(eval(input()))

    a = list(map(int, input().split()))

    b = [0] * N

    b[N // 2] = a[0]

    for i in range(1, N // 2 + 1):

        b[N // 2 - i] = a[i * 2 - 1]

        if i != N // 2 or N % 2 == 1:

            b[N // 2 + i] = a[i * 2]

    if N % 2 == 1:

        b = b[::-1]

    print((" ".join(map(str, b))))


problem_p03673()
