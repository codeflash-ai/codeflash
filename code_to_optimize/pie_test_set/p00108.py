def problem_p00108():
    def FOA(a, count):

        b = [a.count(a[i]) for i in range(len(a))]

        if b == a:

            print(count)

            print((" ".join(map(str, b))))

            return

        else:

            return FOA(b, count + 1)

    while True:

        n = int(eval(input()))

        if n == 0:

            break

        a = list(map(int, input().split()))

        FOA(a, 0)


problem_p00108()
