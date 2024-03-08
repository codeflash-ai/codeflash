def problem_p01713():
    from queue import PriorityQueue

    class State:

        def __init__(self, index, time):

            self.index = index

            self.time = time

        def __lt__(self, state):

            return self.time > state.time

    while 1:

        try:
            n = int(eval(input()))

        except:
            break

        a = list(map(int, input().split()))

        ts = [float("inf")] * n

        vis = [False] * n

        pq = PriorityQueue()

        for i in range(n):

            if a[i] == 0:

                pq.put(State(i, int(1e9)))

        while pq.qsize():

            cur = pq.get()

            if cur.index < 0 or n <= cur.index or vis[cur.index] or cur.time < 0:
                continue

            vis[cur.index] = True

            if a[cur.index] < 0:

                cur.time = min(cur.time, -a[cur.index] - 1)

            ts[cur.index] = cur.time

            pq.put(State(cur.index - 1, cur.time - 1))

            pq.put(State(cur.index + 1, cur.time - 1))

        res = 0

        for i in range(n):

            if a[i] > 0 and vis[i]:

                res += min(a[i], ts[i] + 1)

        print(res)


problem_p01713()
