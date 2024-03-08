def problem_p03666():
    import sys, queue, math, copy, itertools, bisect, collections, heapq

    def main():

        sys.setrecursionlimit(10**7)

        LI = lambda: [int(x) for x in sys.stdin.readline().split()]

        n, a, b, c, d = LI()

        x = int((b - a + d * (n - 1)) / (c + d))

        if x < 0 or x > n - 1:

            print("NO")

            return

        z = c * x - d * (n - 1 - x)

        if z <= b - a <= z + (d - c) * (n - 1):

            print("YES")

        else:

            print("NO")

    if __name__ == "__main__":

        main()


problem_p03666()
