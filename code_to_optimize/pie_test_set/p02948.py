def problem_p02948():
    import queue

    N, M = list(map(int, input().split()))

    AB = [[] for _ in range(M + 1)]

    for n in range(N):

        a, b = list(map(int, input().split()))

        if a > M:

            continue

        AB[a].append(0 - b)

    earn = 0

    q = queue.PriorityQueue()

    for m in range(1, M + 1):

        for job in AB[m]:

            q.put(job)

        if not q.empty():

            earn += q.get()

    print((0 - earn))


problem_p02948()
