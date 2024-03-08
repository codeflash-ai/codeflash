def problem_p03069():
    from heapq import heappush, heappop

    def int_raw():

        return int(eval(input()))

    def ss_raw():

        return input().split()

    def ints_raw():

        return list(map(int, ss_raw()))

    INF = 1 << 29

    def runLength(in_str):

        kukan = 1

        start = 0

        ret = []

        for i in range(1, len(in_str)):

            if in_str[i - 1] != in_str[i]:

                ret.append([in_str[i - 1], kukan, start])

                start = i

                kukan = 0

            kukan += 1

        ret.append([in_str[-1], kukan, start])

        return ret

    N = int_raw()

    S = eval(input())

    dp = {}

    def rec(ss):

        return 0

    def main():

        global S

        ans = INF

        numB = [0] * len(S)

        for i in range(len(S)):

            if i > 0:

                numB[i] = numB[i - 1]

            if S[i] == "#":

                numB[i] += 1

        if numB[len(S) - 1] == 0 or numB[len(S) - 1] == len(S):

            return 0

        for i in range(len(S)):

            buf = numB[i] + len(S) - 1 - i - (numB[len(S) - 1] - numB[i])

            ans = min(ans, buf)

        ans = min(ans, len(S) - numB[len(S) - 1])

        ans = min(ans, numB[len(S) - 1])

        return ans

    print((main()))


problem_p03069()
