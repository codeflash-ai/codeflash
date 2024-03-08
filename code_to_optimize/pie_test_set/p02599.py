def problem_p02599():
    import sys

    read = sys.stdin.read

    readline = sys.stdin.readline

    readlines = sys.stdin.readlines

    sys.setrecursionlimit(10**9)

    INF = 1 << 60

    MOD = 1000000007

    # Reference: https://www.geeksforgeeks.org/queries-number-distinct-elements-subarray/

    # Python3 code to find number of

    # distinct numbers in a subarray

    MAX = 1000001

    # structure to store queries

    class Query:

        def __init__(self, l, r, idx):

            self.l = l

            self.r = r

            self.idx = idx

    # updating the bit array

    def update(idx, val, bit, n):

        while idx <= n:

            bit[idx] += val

            idx += idx & -idx

    # querying the bit array

    def query(idx, bit, n):

        summ = 0

        while idx:

            summ += bit[idx]

            idx -= idx & -idx

        return summ

    def answeringQueries(arr, n, queries, q):

        # initialising bit array

        bit = [0] * (n + 1)

        # holds the rightmost index of

        # any number as numbers of a[i]

        # are less than or equal to 10^6

        last_visit = [-1] * MAX

        # answer for each query

        ans = [0] * q

        query_counter = 0

        for i in range(n):

            # If last visit is not -1 update -1 at the

            # idx equal to last_visit[arr[i]]

            if last_visit[arr[i]] != -1:

                update(last_visit[arr[i]] + 1, -1, bit, n)

            # Setting last_visit[arr[i]] as i and

            # updating the bit array accordingly

            last_visit[arr[i]] = i

            update(i + 1, 1, bit, n)

            # If i is equal to r of any query store answer

            # for that query in ans[]

            while query_counter < q and queries[query_counter].r == i:

                ans[queries[query_counter].idx] = query(
                    queries[query_counter].r + 1, bit, n
                ) - query(queries[query_counter].l, bit, n)

                query_counter += 1

        # print answer for each query

        for i in range(q):

            print((ans[i]))

    # This code is contributed by

    # sanjeev2552

    def main():

        N, Q = list(map(int, readline().split()))

        C = list(map(int, readline().split()))

        (*LR,) = list(map(int, read().split()))

        queries = [0] * Q

        for i, (l, r) in enumerate(zip(*[iter(LR)] * 2)):

            queries[i] = Query(l - 1, r - 1, i)

        queries.sort(key=lambda x: x.r)

        answeringQueries(C, N, queries, Q)

        return

    if __name__ == "__main__":

        main()


problem_p02599()
