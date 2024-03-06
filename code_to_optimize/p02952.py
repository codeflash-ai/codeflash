n = int(eval(input()))





def countKeta(num):

    count = 1

    while num / 10 >= 1:

        count += 1

        num = num // 10

    return count





count = 0



for i in range(1, n+1):

    if(countKeta(i) % 2 == 1):

        count += 1



print(count)