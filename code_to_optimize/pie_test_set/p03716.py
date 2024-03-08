def problem_p03716():
    from heapq import *

    N = int(eval(input()))

    A = list(map(int, input().split()))

    A_indexed = [(a, i) for i, a in enumerate(A)]

    def scores():

        # initial N:2N

        selected_left = A[:N]  # min heap

        heapify(selected_left)

        l_score = sum(selected_left)

        temp = sorted(A_indexed[N:])

        selected_right_set = set(temp[:N])

        unselected_right = temp[N:]  # min heap

        del temp

        r_score = sum([x[0] for x in selected_right_set])

        yield l_score, r_score

        for i in range(N, 2 * N):

            a = A[i]

            # left 使うものを減らす, right 使うものを増やす

            if selected_left[0] < a:

                l_score -= heapreplace(selected_left, a)

                l_score += a

            if (a, i) in selected_right_set:

                r_score -= a

                selected_right_set.remove((a, i))

                n_element = heappop(unselected_right)

                while n_element[1] <= i:

                    n_element = heappop(unselected_right)

                selected_right_set.add(n_element)

                r_score += n_element[0]

            yield l_score, r_score

    # print(list(scores()))

    print((max([x[0] - x[1] for x in scores()])))


problem_p03716()
