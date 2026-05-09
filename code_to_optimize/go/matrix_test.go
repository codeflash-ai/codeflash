package sample

import (
	"math"
	"reflect"
	"testing"
)

func TestMatrixMultiply(t *testing.T) {
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{5, 6}, {7, 8}}
	got := MatrixMultiply(a, b)
	want := [][]float64{{19, 22}, {43, 50}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MatrixMultiply = %v, want %v", got, want)
	}
}

func TestMatrixMultiplyEmpty(t *testing.T) {
	got := MatrixMultiply([][]float64{}, [][]float64{{1}})
	if got != nil {
		t.Errorf("expected nil for empty input, got %v", got)
	}
}

func TestMatrixMultiplyIdentity(t *testing.T) {
	a := [][]float64{{1, 2, 3}, {4, 5, 6}}
	identity := [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}
	got := MatrixMultiply(a, identity)
	if !reflect.DeepEqual(got, a) {
		t.Errorf("A * I = %v, want %v", got, a)
	}
}

func TestMatrixTranspose(t *testing.T) {
	m := [][]float64{{1, 2, 3}, {4, 5, 6}}
	got := MatrixTranspose(m)
	want := [][]float64{{1, 4}, {2, 5}, {3, 6}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MatrixTranspose = %v, want %v", got, want)
	}
}

func TestMatrixTransposeEmpty(t *testing.T) {
	got := MatrixTranspose([][]float64{})
	if got != nil {
		t.Errorf("expected nil for empty input")
	}
}

func TestMatrixAdd(t *testing.T) {
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{5, 6}, {7, 8}}
	got := MatrixAdd(a, b)
	want := [][]float64{{6, 8}, {10, 12}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MatrixAdd = %v, want %v", got, want)
	}
}

func TestMatrixScale(t *testing.T) {
	m := [][]float64{{1, 2}, {3, 4}}
	got := MatrixScale(m, 2.0)
	want := [][]float64{{2, 4}, {6, 8}}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("MatrixScale = %v, want %v", got, want)
	}
}

func TestDotProduct(t *testing.T) {
	got := DotProduct([]float64{1, 2, 3}, []float64{4, 5, 6})
	want := 32.0
	if got != want {
		t.Errorf("DotProduct = %f, want %f", got, want)
	}
}

func TestVectorNorm(t *testing.T) {
	got := VectorNorm([]float64{3, 4})
	want := 5.0
	if got != want {
		t.Errorf("VectorNorm = %f, want %f", got, want)
	}
}

func TestCosineSimilarity(t *testing.T) {
	a := []float64{1, 0}
	b := []float64{0, 1}
	got := CosineSimilarity(a, b)
	if math.Abs(got) > 1e-9 {
		t.Errorf("orthogonal vectors should have cosine similarity 0, got %f", got)
	}

	got = CosineSimilarity([]float64{1, 2, 3}, []float64{1, 2, 3})
	if math.Abs(got-1.0) > 1e-9 {
		t.Errorf("identical vectors should have cosine similarity 1, got %f", got)
	}

	got = CosineSimilarity([]float64{0, 0}, []float64{1, 2})
	if got != 0 {
		t.Errorf("zero vector should give 0, got %f", got)
	}
}

func TestFlattenMatrix(t *testing.T) {
	m := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	got := FlattenMatrix(m)
	want := []float64{1, 2, 3, 4, 5, 6}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("FlattenMatrix = %v, want %v", got, want)
	}
}
