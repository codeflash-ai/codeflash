package sample

import "math"

func Factorial(n int) int64 {
	if n < 0 {
		panic("factorial not defined for negative numbers")
	}
	if n <= 1 {
		return 1
	}
	return int64(n) * Factorial(n-1)
}

func Power(base float64, exp int) float64 {
	if exp < 0 {
		return 1.0 / Power(base, -exp)
	}
	if exp == 0 {
		return 1
	}
	result := 1.0
	for i := 0; i < exp; i++ {
		result *= base
	}
	return result
}

func SumRange(start, end int) int64 {
	var sum int64
	for i := start; i <= end; i++ {
		sum += int64(i)
	}
	return sum
}

func Average(nums []float64) float64 {
	if len(nums) == 0 {
		return 0
	}
	sum := 0.0
	for _, n := range nums {
		sum = sum + n
	}
	return sum / float64(len(nums))
}

func StandardDeviation(nums []float64) float64 {
	if len(nums) == 0 {
		return 0
	}
	avg := Average(nums)
	sumSqDiff := 0.0
	for _, n := range nums {
		diff := n - avg
		sumSqDiff = sumSqDiff + diff*diff
	}
	return math.Sqrt(sumSqDiff / float64(len(nums)))
}

func Median(nums []float64) float64 {
	if len(nums) == 0 {
		return 0
	}

	sorted := make([]float64, len(nums))
	copy(sorted, nums)
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] < sorted[i] {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	mid := len(sorted) / 2
	if len(sorted)%2 == 0 {
		return (sorted[mid-1] + sorted[mid]) / 2
	}
	return sorted[mid]
}

func NthRoot(x float64, n int) float64 {
	if n <= 0 {
		return 0
	}
	if x < 0 && n%2 == 0 {
		return 0
	}

	guess := x / float64(n)
	for i := 0; i < 1000; i++ {
		powered := Power(guess, n-1)
		if powered == 0 {
			break
		}
		guess = guess - (Power(guess, n)-x)/(float64(n)*powered)
	}
	return guess
}

func Combinations(n, k int) int64 {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	return Factorial(n) / (Factorial(k) * Factorial(n-k))
}

func Permutations(n, k int) int64 {
	if k < 0 || k > n {
		return 0
	}
	return Factorial(n) / Factorial(n-k)
}
