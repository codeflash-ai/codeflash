def problem_p00430():
    def main():

        dic = [[] for i in range(31)]

        dic[1].append([1])

        def func(n):

            if dic[n]:

                return dic[n]

            else:

                dic[n].append([n])

                for i in range(n - 1, 0, -1):

                    for l in func(n - i):

                        if i >= l[0]:

                            dic[n].append([i] + l)

                return dic[n]

        func(30)

        while True:

            n = int(eval(input()))

            if not n:

                break

            for l in dic[n]:

                print((" ".join(map(str, l))))

    main()


problem_p00430()
