def problem_p00638():
    while True:

        n = int(eval(input()))

        if n == 0:

            break

        z = sorted([tuple(map(int, input().split())) for _ in range(n)], key=lambda x: x[1])

        cnt = total = 0

        for a, b in z:

            total += a

            if total > b:

                break

            cnt += 1

        print(("Yes" if cnt == n else "No"))


problem_p00638()
