def problem_p00221():
    while True:

        m, n = list(map(int, input().split()))

        if not m:
            break

        a = [True] * m

        j = -1

        for i in range(1, n + 1):

            if sum(a) < 2:
                eval(input())
                continue

            j = (j + 1 + ((a + a)[j + 1 :]).index(True)) % m

            s = eval(input())

            if not i % 15:

                if s == "FizzBuzz":
                    continue

                else:
                    a[j] = False
                    continue

            if not i % 5:

                if s == "Buzz":
                    continue

                else:
                    a[j] = False
                    continue

            if not i % 3:

                if s == "Fizz":
                    continue

                else:
                    a[j] = False
                    continue

            if s != str(i):
                a[j] = False
                continue

        print((" ".join(str(e + 1) for e in range(m) if a[e])))


problem_p00221()
