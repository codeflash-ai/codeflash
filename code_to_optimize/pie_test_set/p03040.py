def problem_p03040():
    import sys, math, collections, heapq, itertools

    from bisect import bisect_left

    F = sys.stdin

    def single_input():
        return F.readline().strip("\n")

    def line_input():
        return F.readline().strip("\n").split()

    def gcd(a, b):

        a, b = max(a, b), min(a, b)

        while a % b > 0:
            a, b = b, a % b

        return b

    def solve():

        Q = int(single_input())

        n, a, b = list(map(int, line_input()))

        left, right = [-a], [a]

        heapq.heapify(left)

        heapq.heapify(right)

        minans = b

        for i in range(1, Q):

            queri = [int(i) for i in line_input()]

            if queri[0] == 1:

                a, b = queri[1:]

                minans += b

                if -1 * left[0] <= a <= right[0]:

                    heapq.heappush(left, -a)

                    heapq.heappush(right, a)

                elif a < -1 * left[0]:

                    minans += (-1 * left[0]) - a

                    heapq.heappush(left, -a)

                    heapq.heappush(left, -a)

                    heapq.heappush(right, -1 * heapq.heappop(left))

                else:

                    minans += a - right[0]

                    heapq.heappush(left, -1 * heapq.heappop(right))

                    heapq.heappush(right, a)

                    heapq.heappush(right, a)

            else:
                print((-1 * left[0], minans))

        return 0

    if __name__ == "__main__":

        solve()


problem_p03040()
