def problem_p01105():
    a = 65280
    b = 61680
    c = 52428
    d = 43690
    e = 65535

    from heapq import heappush, heappop

    base = [a, b, c, d, e, 0]

    Q = [(1, el) for el in base]

    L = {el: 1 for el in base}

    H = []

    get = L.get

    push = H.append

    while Q:

        l, p = heappop(Q)

        if L[p] < l:
            continue

        if l + 1 < get(p ^ e, 17):

            L[p ^ e] = l + 1

            if l + 1 < 16:
                heappush(Q, (l + 1, p ^ e))

        if l + 3 < 16:

            for q, r in H:

                if l + r + 3 <= 16:

                    if l + r + 3 < get(p & q, 17):

                        L[p & q] = l + r + 3

                        if l + r + 3 < 16:
                            heappush(Q, (l + r + 3, p & q))

                    if l + r + 3 < get(p ^ q, 17):

                        L[p ^ q] = l + r + 3

                        if l + r + 3 < 16:
                            heappush(Q, (l + r + 3, p ^ q))

                else:
                    break

        if l < 7:
            push((p, l))

    print(
        *map(
            L.__getitem__,
            eval(
                "e&%s"
                % ",e&".join(
                    open(0).read().replace(*"-~").replace(*"*&").replace(*"1e").split()[:-1]
                )
            ),
        ),
        sep="\n",
    )


problem_p01105()
