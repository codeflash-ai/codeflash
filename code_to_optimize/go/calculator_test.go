package sample

import (
	"math"
	"testing"
)

func TestFactorial(t *testing.T) {
	tests := []struct {
		n    int
		want int64
	}{
		{0, 1},
		{1, 1},
		{5, 120},
		{10, 3628800},
	}

	for _, tc := range tests {
		got := Factorial(tc.n)
		if got != tc.want {
			t.Errorf("Factorial(%d) = %d, want %d", tc.n, got, tc.want)
		}
	}
}

func TestFactorialPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative input")
		}
	}()
	Factorial(-1)
}

func TestPower(t *testing.T) {
	tests := []struct {
		base float64
		exp  int
		want float64
	}{
		{2, 10, 1024},
		{3, 0, 1},
		{5, 1, 5},
		{2, -1, 0.5},
	}

	for _, tc := range tests {
		got := Power(tc.base, tc.exp)
		if math.Abs(got-tc.want) > 1e-9 {
			t.Errorf("Power(%f, %d) = %f, want %f", tc.base, tc.exp, got, tc.want)
		}
	}
}

func TestSumRange(t *testing.T) {
	if got := SumRange(1, 100); got != 5050 {
		t.Errorf("SumRange(1,100) = %d, want 5050", got)
	}
	if got := SumRange(5, 5); got != 5 {
		t.Errorf("SumRange(5,5) = %d, want 5", got)
	}
}

func TestAverage(t *testing.T) {
	got := Average([]float64{1, 2, 3, 4, 5})
	if got != 3.0 {
		t.Errorf("Average = %f, want 3.0", got)
	}

	got = Average([]float64{})
	if got != 0 {
		t.Errorf("Average empty = %f, want 0", got)
	}
}

func TestStandardDeviation(t *testing.T) {
	got := StandardDeviation([]float64{2, 4, 4, 4, 5, 5, 7, 9})
	if math.Abs(got-2.0) > 0.01 {
		t.Errorf("StandardDeviation = %f, want ~2.0", got)
	}
}

func TestMedian(t *testing.T) {
	got := Median([]float64{3, 1, 2})
	if got != 2.0 {
		t.Errorf("Median odd = %f, want 2.0", got)
	}

	got = Median([]float64{4, 1, 3, 2})
	if got != 2.5 {
		t.Errorf("Median even = %f, want 2.5", got)
	}

	got = Median([]float64{})
	if got != 0 {
		t.Errorf("Median empty = %f, want 0", got)
	}
}

func TestNthRoot(t *testing.T) {
	got := NthRoot(27, 3)
	if math.Abs(got-3.0) > 1e-6 {
		t.Errorf("NthRoot(27,3) = %f, want 3.0", got)
	}

	got = NthRoot(16, 4)
	if math.Abs(got-2.0) > 1e-6 {
		t.Errorf("NthRoot(16,4) = %f, want 2.0", got)
	}
}

func TestCombinations(t *testing.T) {
	tests := []struct {
		n, k int
		want int64
	}{
		{5, 2, 10},
		{10, 3, 120},
		{5, 0, 1},
		{5, 5, 1},
		{3, 5, 0},
	}

	for _, tc := range tests {
		got := Combinations(tc.n, tc.k)
		if got != tc.want {
			t.Errorf("Combinations(%d,%d) = %d, want %d", tc.n, tc.k, got, tc.want)
		}
	}
}

func TestPermutations(t *testing.T) {
	tests := []struct {
		n, k int
		want int64
	}{
		{5, 2, 20},
		{5, 0, 1},
		{3, 5, 0},
	}

	for _, tc := range tests {
		got := Permutations(tc.n, tc.k)
		if got != tc.want {
			t.Errorf("Permutations(%d,%d) = %d, want %d", tc.n, tc.k, got, tc.want)
		}
	}
}
