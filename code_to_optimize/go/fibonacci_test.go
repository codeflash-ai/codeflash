package sample

import (
	"reflect"
	"testing"
)

func TestFibonacci(t *testing.T) {
	tests := []struct {
		n    int
		want int64
	}{
		{0, 0},
		{1, 1},
		{2, 1},
		{5, 5},
		{10, 55},
		{20, 6765},
	}

	for _, tc := range tests {
		got := Fibonacci(tc.n)
		if got != tc.want {
			t.Errorf("Fibonacci(%d) = %d, want %d", tc.n, got, tc.want)
		}
	}
}

func TestFibonacciPanicsOnNegative(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for negative input")
		}
	}()
	Fibonacci(-1)
}

func TestIsFibonacci(t *testing.T) {
	fibs := []int64{0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55}
	for _, f := range fibs {
		if !IsFibonacci(f) {
			t.Errorf("IsFibonacci(%d) = false, want true", f)
		}
	}

	nonFibs := []int64{4, 6, 7, 9, 10, 22}
	for _, f := range nonFibs {
		if IsFibonacci(f) {
			t.Errorf("IsFibonacci(%d) = true, want false", f)
		}
	}

	if IsFibonacci(-1) {
		t.Error("IsFibonacci(-1) should be false")
	}
}

func TestFibonacciSequence(t *testing.T) {
	got := FibonacciSequence(7)
	want := []int64{0, 1, 1, 2, 3, 5, 8}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("FibonacciSequence(7) = %v, want %v", got, want)
	}

	got = FibonacciSequence(0)
	if len(got) != 0 {
		t.Errorf("FibonacciSequence(0) should be empty, got %v", got)
	}
}

func TestFibonacciIndex(t *testing.T) {
	tests := []struct {
		num  int64
		want int
	}{
		{0, 0},
		{1, 1},
		{5, 5},
		{8, 6},
		{55, 10},
		{4, -1},
		{-1, -1},
	}

	for _, tc := range tests {
		got := FibonacciIndex(tc.num)
		if got != tc.want {
			t.Errorf("FibonacciIndex(%d) = %d, want %d", tc.num, got, tc.want)
		}
	}
}

func TestSumFibonacci(t *testing.T) {
	tests := []struct {
		n    int
		want int64
	}{
		{0, 0},
		{1, 0},
		{5, 7},
		{7, 20},
	}

	for _, tc := range tests {
		got := SumFibonacci(tc.n)
		if got != tc.want {
			t.Errorf("SumFibonacci(%d) = %d, want %d", tc.n, got, tc.want)
		}
	}
}

func TestFibonacciUpTo(t *testing.T) {
	got := FibonacciUpTo(10)
	want := []int64{0, 1, 1, 2, 3, 5, 8}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("FibonacciUpTo(10) = %v, want %v", got, want)
	}

	got = FibonacciUpTo(0)
	if len(got) != 0 {
		t.Errorf("FibonacciUpTo(0) should be empty")
	}
}

func TestAreConsecutiveFibonacci(t *testing.T) {
	if !AreConsecutiveFibonacci(5, 8) {
		t.Error("5 and 8 are consecutive fibonacci numbers")
	}
	if !AreConsecutiveFibonacci(8, 5) {
		t.Error("8 and 5 are consecutive fibonacci numbers")
	}
	if AreConsecutiveFibonacci(5, 13) {
		t.Error("5 and 13 are not consecutive fibonacci numbers")
	}
	if AreConsecutiveFibonacci(4, 5) {
		t.Error("4 is not a fibonacci number")
	}
}
