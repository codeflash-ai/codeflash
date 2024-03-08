def problem_p03828():
    from collections import Counter

    N = int(eval(input()))

    div = []

    for i in range(2, N + 1):

        while i != 1:

            for d in range(2, i + 1):

                while i % d == 0:

                    i = i // d

                    div.append(d)

    C = Counter(div).most_common()

    Ans = 1

    for k, v in C:

        Ans = Ans * (v + 1) % (10**9 + 7)

    print(Ans)


problem_p03828()
