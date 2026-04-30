package sample

import "math"

func MatrixMultiply(a, b [][]float64) [][]float64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	rows := len(a)
	cols := len(b[0])
	inner := len(b)

	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum := 0.0
			for k := 0; k < inner; k++ {
				sum = sum + a[i][k]*b[k][j]
			}
			result[i][j] = sum
		}
	}
	return result
}

func MatrixTranspose(m [][]float64) [][]float64 {
	if len(m) == 0 {
		return nil
	}

	rows := len(m)
	cols := len(m[0])

	result := make([][]float64, cols)
	for i := range result {
		result[i] = make([]float64, rows)
	}

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j][i] = m[i][j]
		}
	}
	return result
}

func MatrixAdd(a, b [][]float64) [][]float64 {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}

	rows := len(a)
	cols := len(a[0])

	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = a[i][j] + b[i][j]
		}
	}
	return result
}

func MatrixScale(m [][]float64, scalar float64) [][]float64 {
	if len(m) == 0 {
		return nil
	}

	rows := len(m)
	cols := len(m[0])

	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] * scalar
		}
	}
	return result
}

func DotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := 0; i < len(a); i++ {
		sum = sum + a[i]*b[i]
	}
	return sum
}

func VectorNorm(v []float64) float64 {
	sum := 0.0
	for _, val := range v {
		sum = sum + val*val
	}
	return math.Sqrt(sum)
}

func CosineSimilarity(a, b []float64) float64 {
	dot := DotProduct(a, b)
	normA := VectorNorm(a)
	normB := VectorNorm(b)
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (normA * normB)
}

func FlattenMatrix(m [][]float64) []float64 {
	var result []float64
	for _, row := range m {
		for _, val := range row {
			result = append(result, val)
		}
	}
	return result
}
