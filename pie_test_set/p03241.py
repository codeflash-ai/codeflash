def problem_p03241(input_data):
    N, M = list(map(int, input_data.split()))

    if N == 1:

        nmax = M

    else:

        nmax = 1

        for i in range(2, int(M**0.5) + 1):

            if M % i == 0 and M // i >= N:

                nmax = max(nmax, i)

            if M % i == 0 and i >= N:

                nmax = max(nmax, M // i)

    return nmax
