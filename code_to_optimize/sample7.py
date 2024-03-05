def main():
    while True:
        try:
            n = int(input())
        except EOFError:
            break
        n = ((n + 1) * n) // 2
        print(n)

if __name__ == "__main__":
    main()
