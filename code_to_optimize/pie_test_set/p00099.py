def problem_p00099():
    import queue

    n, q = list(map(int, input().split()))

    p = queue.PriorityQueue()

    s = [0] * -~n

    for _ in [0] * q:

        a, v = list(map(int, input().split()))

        s[a] += v
        p.put((-s[a], a))

        while 1:

            t = p.get()

            if -t[0] == s[t[1]]:
                print((t[1], -t[0]))
                p.put(t)
                break


problem_p00099()
