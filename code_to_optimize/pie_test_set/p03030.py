def problem_p03030():
    def main():

        N = int(eval(input()))

        sp_lst = [0 for _ in range(N)]

        for i in range(N):

            S, P = input().split()

            sp_lst[i] = [i + 1, S, int(P)]

        sp_lst.sort(key=lambda x: x[2], reverse=True)

        sp_lst.sort(key=lambda x: x[1])

        for i, _, _ in sp_lst:

            print(i)

    if __name__ == "__main__":

        main()


problem_p03030()
