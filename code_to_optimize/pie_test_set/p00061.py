def problem_p00061():
    import itertools

    from operator import itemgetter

    nums = {}

    while True:

        p, s = list(map(int, input().split(",")))

        if p == 0 and s == 0:
            break

        if not s in nums:

            nums[s] = []

        nums[s].append(p)

    while True:

        try:

            q = int(eval(input()))

            ans = len(nums)

            for ns in list(nums.items()):

                if q in ns[1]:

                    print(ans)

                    break

                ans -= 1

        except:

            break


problem_p00061()
