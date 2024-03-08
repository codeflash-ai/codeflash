def problem_p00120():
    import sys

    for s in sys.stdin:

        a = list(map(int, s.split(" ")))

        w = a[0]

        a = sorted(a[1:])

        A = []

        try:

            while a:

                A += [a.pop(0)]

                A = [a.pop()] + A

                A += [a.pop()]

                A = [a.pop(0)] + A

        except:
            pass

        a = A[0] + A[-1]

        for i in range(len(A) - 1):
            a += 2 * (A[i] * A[i + 1]) ** 0.5

        print(["OK", "NA"][a > w])


problem_p00120()
