def problem_p02268():
    N = int(eval(input()))

    S = list(map(int, input().split()))

    Q = int(eval(input()))

    T = list(map(int, input().split()))

    def binary_search(target_num, head, tail):

        idx = int((head + tail) / 2)

        if target_num == S[idx]:

            return True

        elif target_num < S[idx]:

            if head == idx:

                return False

            return binary_search(target_num, head, idx - 1)

        else:

            if tail == idx:

                return False

            return binary_search(target_num, idx + 1, tail)

    match_count = 0

    for target_num in T:

        if binary_search(target_num, 0, len(S) - 1):

            match_count += 1

    print((str(match_count)))


problem_p02268()
