def problem_p03214():
    def main():

        n = int(eval(input()))

        a_list = list(map(int, input().split()))

        mean_a = sum(a_list) / n

        diff_list = [abs(a - mean_a) for a in a_list]

        min_a = min(diff_list)

        print((diff_list.index(min_a)))

    if __name__ == "__main__":

        main()


problem_p03214()
