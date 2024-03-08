def problem_p02259():
    def bubble_sort(A):

        count = 0

        for i in reversed(list(range(len(A)))):

            for j in range(i):

                if A[j] > A[j + 1]:

                    temp = A[j]

                    A[j] = A[j + 1]

                    A[j + 1] = temp

                    count += 1

        return count

    N = int(eval(input()))

    A = list(map(int, input().split()))

    count = bubble_sort(A)

    print((" ".join(map(str, A))))

    print(count)


problem_p02259()
