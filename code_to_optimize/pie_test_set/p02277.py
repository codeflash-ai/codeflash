def problem_p02277():
    from collections import namedtuple

    Card = namedtuple("Card", "suit value")

    def partition(A, p, r):

        x = A[r]

        i = p

        for j in range(p, r):

            if A[j].value <= x.value:

                A[i], A[j] = A[j], A[i]

                i += 1

        A[i], A[r] = A[r], A[i]

        return i

    def quickSort(A, p, r):

        if p < r:

            q = partition(A, p, r)

            quickSort(A, p, q - 1)

            quickSort(A, q + 1, r)

    def merge(L, R):

        global cnt

        n = len(L) + len(R)

        A = []

        i = j = 0

        L.append(Card("X", -1))

        R.append(Card("X", -1))

        for _ in range(n):

            if L[i].value > R[j].value:

                A.append(L[i])

                i += 1

            else:

                A.append(R[j])

                j += 1

        return A

    def mergeSort(A):

        if len(A) == 1:
            return A

        m = len(A) // 2

        return merge(mergeSort(A[:m]), mergeSort(A[m:]))

    if __name__ == "__main__":

        n = int(eval(input()))

        Co = list([Card(X[0], int(X[1])) for X in [input().split() for _ in range(n)]])

        Cq = Co[:]

        Cm = mergeSort(Co[:])

        quickSort(Cq, 0, n - 1)

        print(("Stable" if Cq == Cm[::-1] else "Not stable"))

        for c in Cq:
            print((c.suit, c.value))


problem_p02277()
