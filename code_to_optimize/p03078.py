# -*- coding: utf-8 -*-

x, y, z, k = list(map(int, input().split()))

a = list(map(int, input().split()))

b = list(map(int, input().split()))

c = list(map(int, input().split()))


a.sort(reverse=True)

b.sort(reverse=True)

c.sort(reverse=True)


abc = []


for i, _a in enumerate(a):
    for j, _b in enumerate(b):
        if (i + 1) * (j + 1) > k:
            break

        for l, _c in enumerate(c):
            if (i + 1) * (j + 1) * (l + 1) > k:
                break

            abc.append(_a + _b + _c)


abc.sort(reverse=True)

for x in abc[:k]:
    print(x)
