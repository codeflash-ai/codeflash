def main():
    n = int(input())
    a = [int(input()) for _ in range(n)]
    a.sort(reverse=True)

    ma = 0
    num = 0
    i = 0
    while i < n - 1:
        if a[i] == a[i + 1]:
            if num == 0:
                ma = a[i]
            else:
                ma *= a[i]
            num += 1
            i += 1
        if num >= 2:
            break
        i += 1

    if num >= 2:
        print(ma)
    else:
        print(0)

if __name__ == "__main__":
    main()
