def problem_p02260():
    def main():

        N = int(eval(input()))

        A = list(map(int, input().split(" ")))

        A, count = selectionSort(A, N)

        print((" ".join([str(a) for a in A])))

        print(count)

    def selectionSort(A, N):

        count = 0

        for i in range(0, N - 1):

            minj = i

            for j in range(i, N):

                if A[j] < A[minj]:

                    minj = j

            if i != minj:

                A[i], A[minj] = A[minj], A[i]

                count += 1

        return A, count

    if __name__ == "__main__":

        main()


problem_p02260()
