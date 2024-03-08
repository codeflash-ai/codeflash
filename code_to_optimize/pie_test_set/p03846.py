def problem_p03846():
    n = int(eval(input()))

    a = list(map(int, input().split()))

    ans = 0

    if n % 2 == 1:

        dummy = [0]

        for i in range(1, 1 + (len(a) - 1) // 2):

            dummy.append(2 * i)

            dummy.append(2 * i)

        if sorted(a) == dummy:

            ans = 2 ** ((len(a) - 1) // 2)

    else:

        dummy = []

        for i in range(1, 1 + (len(a)) // 2):

            dummy.append(2 * i - 1)

            dummy.append(2 * i - 1)

        if sorted(a) == dummy:

            ans = 2 ** ((len(a)) // 2)

    print((ans % (10**9 + 7)))


problem_p03846()
