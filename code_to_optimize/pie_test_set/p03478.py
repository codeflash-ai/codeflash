def problem_p03478(input_data):
    N, A, B = list(map(int, input_data.split()))

    ans = 0

    for i in range(1, N + 1):

        if A <= sum(list(map(int, str(i)))) <= B:

            ans += i

    return ans
