def problem_p03032():
    from collections import deque

    n, k = list(map(int, input().split()))

    V = list(map(int, input().split()))

    ans = 0

    for t in range(k // 2 + 1):

        hand = deque(V[: k - t])

        rest = deque(V[k - t :])

        cur_sum = sum(hand)

        for j in range(k - t + 1):

            put_back = 0

            rev_hand = sorted(hand)

            for i in range(t):

                jewel = rev_hand[i]

                if jewel < 0:

                    put_back -= jewel

                else:

                    break

            candidate = cur_sum + put_back

            ans = max(candidate, ans)

            to_rest = hand.pop()

            rest.appendleft(to_rest)

            to_hand = rest.pop()

            hand.appendleft(to_hand)

            cur_sum += to_hand - to_rest

    print(ans)


problem_p03032()
