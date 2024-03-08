def problem_p02239():
    import queue, sys

    N = int(eval(input()))

    M = [list(map(int, e.split()[2:])) for e in sys.stdin]

    q = queue.Queue()
    q.put(0)

    d = [-1] * N
    d[0] = 0

    while q.qsize() > 0:

        u = q.get()

        for v in M[u]:

            v -= 1

            if d[v] < 0:
                d[v] = d[u] + 1
                q.put(v)

    for i in range(N):
        print((i + 1, d[i]))


problem_p02239()
