def problem_p02289():
    from sys import stdin

    readline = stdin.readline

    import heapq

    def main():

        heap = []

        while True:

            line = readline()

            if line[2] == "t":

                print((heapq.heappop(heap)[1]))

            elif line[2] == "s":

                n = int(line.split()[1])

                heapq.heappush(heap, (-n, n))

            else:

                break

    main()


problem_p02289()
