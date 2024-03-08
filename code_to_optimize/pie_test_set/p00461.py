def problem_p00461():
    while True:

        n = eval(input())

        if n == 0:

            break

        m = eval(input())

        s = input()

        cur, score = 0, 0

        for c in s:

            if cur <= 2 * n:

                if cur % 2 == 0:

                    cur = cur + 1 if c == "I" else 0

                else:

                    cur = 1 if c == "I" else cur + 1

            elif cur == 2 * n + 1:

                score += 1

                cur = 1 if c == "I" else 2 * n

        print(score)


problem_p00461()
