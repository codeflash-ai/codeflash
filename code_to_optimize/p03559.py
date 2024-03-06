N = int(eval(input()))

A = list(map(int,input().split()))

B = list(map(int,input().split()))

C = list(map(int,input().split()))

A.sort()

C.sort()



ans = 0

from bisect import bisect, bisect_left

for b in B:

    i = bisect_left(A, b)

    j = bisect(C, b)

    ans += i * (N-j)

print(ans)