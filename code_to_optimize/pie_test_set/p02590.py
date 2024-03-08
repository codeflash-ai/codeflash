def problem_p02590():
    import numpy as np

    if __name__ == "__main__":

        N = int(eval(input()))

        A = list(map(int, input().split()))

        P = 200_003

        rootP = 2

        order = [1]

        x = rootP

        while x != 1:

            order.append(x)

            x *= rootP

            x %= P

        ORDER_SIZE = len(order)  # 200_002

        where = np.zeros(P, int)

        for i in range(ORDER_SIZE):

            where[order[i]] = i

        prevCnt = np.zeros(P, int)

        resCnt = np.zeros(P, int)

        for n in range(N):

            if A[n] != 0:

                prevCnt[where[A[n]]] += 1

                resCnt[(A[n] * A[n]) % P] -= 1

        N_ = 1

        while N_ < 2 * len(prevCnt):
            N_ *= 2

        nf = np.zeros(N_, int)

        nf[: len(prevCnt)] = prevCnt

        nf = np.fft.rfft(nf)

        fftRes = np.rint(np.fft.irfft(nf * nf))

        for r in range(len(fftRes)):

            if fftRes[r] != 0:

                resCnt[order[r % ORDER_SIZE]] += fftRes[r]

        ans = 0

        for p in range(P):

            if resCnt[p] != 0:

                ans += (resCnt[p] // 2) * p

        print(ans)


problem_p02590()
