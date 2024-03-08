def problem_p02563():
    import numpy as np

    from scipy import signal

    MOD = 998244353

    N, M = list(map(int, input().split()))

    As = list(map(int, input().split()))

    Bs = list(map(int, input().split()))

    def divideArr3(As, numD):

        arrA = np.array(As, dtype=np.int64)

        mask = (1 << numD) - 1

        arrA0 = arrA & mask

        arrA1 = (arrA >> numD) & mask

        arrA2 = arrA >> (2 * numD)

        return arrA2, arrA1, arrA0

    def convolveArr(arrA, arrB):

        arrC = np.rint(signal.fftconvolve(arrA, arrB)).astype(np.int64)

        return arrC

    def convolveMOD(As, Bs, MOD):

        numD = 10

        div = 1 << numD

        arrA2, arrA1, arrA0 = divideArr3(As, numD)

        arrB2, arrB1, arrB0 = divideArr3(Bs, numD)

        arrC4 = convolveArr(arrA2, arrB2) % MOD

        arrC2 = convolveArr(arrA1, arrB1) % MOD

        arrC0 = convolveArr(arrA0, arrB0) % MOD

        arrC3 = (convolveArr(arrA2 + arrA1, arrB2 + arrB1) % MOD - arrC4 - arrC2) % MOD

        arrC1 = (convolveArr(arrA1 + arrA0, arrB1 + arrB0) % MOD - arrC2 - arrC0) % MOD

        arrC2 += (convolveArr(arrA2 + arrA0, arrB2 + arrB0) % MOD - arrC4 - arrC0) % MOD

        arrC2 %= MOD

        div2 = div * div % MOD

        div3 = div2 * div % MOD

        div4 = div3 * div % MOD

        arrAns = (
            (arrC4 * div4) % MOD
            + (arrC3 * div3) % MOD
            + (arrC2 * div2) % MOD
            + (arrC1 * div) % MOD
            + arrC0
        ) % MOD

        return arrAns.tolist()

    anss = convolveMOD(As, Bs, MOD)

    print((" ".join(map(str, anss))))


problem_p02563()
