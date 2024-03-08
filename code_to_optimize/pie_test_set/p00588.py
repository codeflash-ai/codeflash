def problem_p00588():
    pre = [
        ((0, 1, 2), (1, 0, 1), (2, 1, 0)),
        ((3, 2, 2), (2, 1, 1), (2, 1, 1)),
        ((1, 1, 2), (1, 1, 2), (2, 2, 3)),
        ((3, 2, 2), (2, 2, 2), (2, 2, 3)),
    ]

    q = int(eval(input()))

    for _ in range(q):

        n = int(eval(input()))

        books = [c == "Y" for c in eval(input())]

        books = (
            [(False, False)]
            + list(map(lambda u, l: (u, l), books[: 2 * n], books[2 * n :]))
            + [(False, False)]
        )

        shelves = [
            int(u1 or u2) * 2 + int(l1 or l2) for (u1, l1), (u2, l2) in zip(*[iter(books)] * 2)
        ]

        ans = [0, 1, 2]

        for key in shelves:

            new_ans = [min(a + c for a, c in zip(ans, costs)) for costs in pre[key]]

            ans = new_ans

        print((ans[0] + n))


problem_p00588()
