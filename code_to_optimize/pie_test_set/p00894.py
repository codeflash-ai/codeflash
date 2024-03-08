def problem_p00894():
    while 1:

        n = int(input())

        if n == 0:
            break

        exist = set([])

        time = [0] * 1000

        bless = [0] * 1000

        for loop in range(n):

            md, hm, io, p = input().split()

            h, m = list(map(int, hm.split(":")))

            t = 60 * h + m

            p = int(p)

            if io == "I":

                time[p] = t

                exist.add(p)

            else:

                exist.remove(p)

                if p == 0:

                    for i in exist:
                        bless[i] += t - max(time[p], time[i])

                elif 0 in exist:

                    bless[p] += t - max(time[0], time[p])

        print(max(bless))


problem_p00894()
