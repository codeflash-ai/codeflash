package calculator

import "math"

// Add returns the sum of two integers.
func Add(a, b int) int {
	return a + b
}

func Subtract(a, b int) int {
	return a - b
}

// unexported function
func multiply(a, b int) int {
	return a * b
}

// no return type
func init() {
	// package initialization
}

func Fibonacci(n int) int {
	if n <= 1 {
		return n
	}
	return Fibonacci(n-1) + Fibonacci(n-2)
}

// Hypotenuse calculates the hypotenuse of a right triangle.
func Hypotenuse(a, b float64) float64 {
	return math.Sqrt(a*a + b*b)
}

type Calculator struct {
	Result float64
}

// AddFloat adds a value to the calculator result.
func (c *Calculator) AddFloat(val float64) float64 {
	c.Result += val
	return c.Result
}

func (c Calculator) GetResult() float64 {
	return c.Result
}

// Reset zeroes the calculator.
func (c *Calculator) Reset() {
	c.Result = 0
}
