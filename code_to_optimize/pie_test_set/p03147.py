def problem_p03147():
    N = int(eval(input()))

    lst = [int(s) for s in input().split()]

    def suc(ls):

        val_ls = ls

        i = 0

        while i < N and val_ls[i] == 0:

            i += 1

        if i == N:

            return []

        else:

            while i < N and val_ls[i] != 0:

                val_ls[i] -= 1

                i += 1

            return val_ls

    ans = 0

    while lst != []:

        ans += 1

        lst = suc(lst)

    print((ans - 1))


problem_p03147()
