def problem_p03296():
    import random

    N = int(eval(input()))

    a = [int(ai) for ai in input().split()]

    random.seed(42)

    count = 0

    for i in range(0, N - 1):

        if a[i] == a[i + 1]:

            count += 1

        while a[i] == a[i + 1]:

            a[i + 1] = random.randint(1, N)

            if i < N - 2:

                while a[i + 1] == a[i + 2]:

                    a[i + 1] = random.randint(1, N)

    print(count)


problem_p03296()
