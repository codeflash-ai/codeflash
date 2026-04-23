package calculator

import "testing"

func TestAdd(t *testing.T) {
	result := Add(2, 3)
	if result != 5 {
		t.Errorf("Add(2, 3) = %d; want 5", result)
	}
}

func TestSubtract(t *testing.T) {
	result := Subtract(5, 3)
	if result != 2 {
		t.Errorf("Subtract(5, 3) = %d; want 2", result)
	}
}

func TestFibonacci(t *testing.T) {
	tests := []struct {
		input    int
		expected int
	}{
		{0, 0},
		{1, 1},
		{10, 55},
	}
	for _, tt := range tests {
		result := Fibonacci(tt.input)
		if result != tt.expected {
			t.Errorf("Fibonacci(%d) = %d; want %d", tt.input, result, tt.expected)
		}
	}
}
