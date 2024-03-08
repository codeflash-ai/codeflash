def problem_p02812():
    N = int(eval(input()))

    S = eval(input())

    answer = 0

    for i in range(N - 2):

        if S[i : i + 3] == "ABC":

            answer += 1

    print(answer)


problem_p02812()
