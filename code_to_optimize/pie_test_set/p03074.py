def problem_p03074():
    N, K = list(map(int, input().split()))

    S = eval(input())

    nums = []

    now = 1

    cnt = 0

    for i in range(N):

        if S[i] == str(now):

            cnt += 1

        else:

            nums.append(cnt)

            now = 1 - now

            cnt = 1

    if cnt != 0:

        nums.append(cnt)

    if len(nums) % 2 == 0:

        nums.append(0)

    add = 2 * K + 1

    ans = 0

    left = 0

    right = 0

    tmp = 0

    for i in range(0, len(nums), 2):

        next_left = i

        next_right = min(i + add, len(nums))

        while next_left > left:

            tmp -= nums[left]

            left += 1

        while next_right > right:

            tmp += nums[right]

            right += 1

        ans = max(ans, tmp)

    print(ans)


problem_p03074()
