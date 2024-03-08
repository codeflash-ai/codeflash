def problem_p02264():
    class Queue:

        def __init__(self, values):

            self.values = values

        def empty(self):

            if len(self.values) == 0:

                return True

            else:

                return False

        def enqueue(self, v):

            self.values.append(v)

        def dequeue(self):

            if len(self.values) <= 0:

                raise

            else:

                v = self.values[0]

                del self.values[0]

                return v

    n, q = list(map(int, input().split(" ")))

    processes = []

    for i in range(n):

        n, t = input().split(" ")

        processes.append((n, int(t)))

    queue = Queue(processes)

    clock = 0

    done = []

    while not queue.empty():

        p = queue.dequeue()

        n = p[0]

        t = p[1]

        if t <= q:

            clock += t

            done.append((n, clock))

        else:

            queue.enqueue((n, t - q))

            clock += q

    for p in done:

        print(p[0], p[1])


problem_p02264()
