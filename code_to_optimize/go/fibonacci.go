package sample

import "math"

func Fibonacci(n int) int64 {
	if n < 0 {
		panic("fibonacci not defined for negative numbers")
	}
	if n <= 1 {
		return int64(n)
	}
	return Fibonacci(n-1) + Fibonacci(n-2)
}

func IsFibonacci(num int64) bool {
	if num < 0 {
		return false
	}
	check1 := 5*num*num + 4
	check2 := 5*num*num - 4
	return isPerfectSquare(check1) || isPerfectSquare(check2)
}

func isPerfectSquare(n int64) bool {
	if n < 0 {
		return false
	}
	sqrt := int64(math.Sqrt(float64(n)))
	return sqrt*sqrt == n
}

func FibonacciSequence(n int) []int64 {
	if n < 0 {
		panic("n must be non-negative")
	}
	if n == 0 {
		return []int64{}
	}

	result := make([]int64, n)
	for i := 0; i < n; i++ {
		result[i] = Fibonacci(i)
	}
	return result
}

func FibonacciIndex(fibNum int64) int {
	if fibNum < 0 {
		return -1
	}
	if fibNum == 0 {
		return 0
	}
	if fibNum == 1 {
		return 1
	}

	for index := 2; index <= 50; index++ {
		fib := Fibonacci(index)
		if fib == fibNum {
			return index
		}
		if fib > fibNum {
			return -1
		}
	}
	return -1
}

func SumFibonacci(n int) int64 {
	if n <= 0 {
		return 0
	}
	var sum int64
	for i := 0; i < n; i++ {
		sum += Fibonacci(i)
	}
	return sum
}

func FibonacciUpTo(limit int64) []int64 {
	var result []int64
	if limit <= 0 {
		return result
	}

	for index := 0; index <= 50; index++ {
		fib := Fibonacci(index)
		if fib >= limit {
			break
		}
		result = append(result, fib)
	}
	return result
}

func AreConsecutiveFibonacci(a, b int64) bool {
	if !IsFibonacci(a) || !IsFibonacci(b) {
		return false
	}
	indexA := FibonacciIndex(a)
	indexB := FibonacciIndex(b)
	diff := indexA - indexB
	if diff < 0 {
		diff = -diff
	}
	return diff == 1
}
