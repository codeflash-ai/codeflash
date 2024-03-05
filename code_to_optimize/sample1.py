def binary(r, k):
    l = 1
    while l <= r:
        mid = (l + r) // 2
        if a[mid] <= k:
            l = mid + 1
        else:
            r = mid - 1
    return l - 1

def main():
    n, m = map(int, input().split())
    stack = [0] * (n + 1)
    t = [0] * (n + 1)
    ans = [0] * (n + 1)
    a = [0] * (n + 1)
    top = 0
    stack[top] = n
    for i in range(1, m + 1):
        x = int(input())
        while x <= stack[top]:
            top -= 1
        top += 1
        stack[top] = x
    m = top
    for i in range(1, top + 1):
        a[i] = stack[i]
    t[m] = 1
    for i in range(m, 0, -1):
        k = a[i]
        p = binary(i - 1, k)
        while p:
            t[p] += k // a[p] * t[i]
            k %= a[p]
            p = binary(p - 1, k)
        ans[k] += t[i]
    for i in range(n, 0, -1):
        ans[i] += ans[i + 1]
    for i in range(1, n + 1):
        print(ans[i])

if __name__ == "__main__":
    main()
