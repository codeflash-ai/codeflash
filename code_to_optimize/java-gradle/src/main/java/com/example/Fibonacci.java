package com.example;

public class Fibonacci {

    public static long fibonacci(int n) {
        if (n < 0) {
            throw new IllegalArgumentException("n must be non-negative");
        }
        if (n <= 1) {
            return n;
        }
        long a = 0L; // F(0)
        long b = 1L; // F(1)
        int highestBit = 31 - Integer.numberOfLeadingZeros(n);
        for (int i = highestBit; i >= 0; --i) {
            // apply fast doubling:
            // c = F(2k) = F(k) * (2*F(k+1) - F(k))
            // d = F(2k+1) = F(k)^2 + F(k+1)^2
            long c = a * (2 * b - a);
            long d = a * a + b * b;
            if (((n >> i) & 1) == 0) {
                a = c;
                b = d;
            } else {
                a = d;
                b = c + d;
            }
        }
        return a;
    }
}
