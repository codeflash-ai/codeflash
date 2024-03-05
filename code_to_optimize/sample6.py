def main():
    s = input()
    l = len(s)

    if l % 2 != 0:
        print("No")
        return

    s1 = s[::-1]
    s1 = s1.translate(str.maketrans('bdpq', 'dbqp'))

    if s1 == s:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()
