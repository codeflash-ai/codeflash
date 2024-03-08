def problem_p00469():
    from itertools import permutations as P

    while True:

        n, k = int(eval(input())), int(eval(input()))

        if k == 0:

            break

        card = [int(eval(input())) for _ in range(n)]

        print((len(set(["".join(map(str, s)) for s in set(P(card, k))]))))


problem_p00469()
