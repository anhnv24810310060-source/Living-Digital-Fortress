package sandbox

// minFloat returns the smaller of two float64 values.
func minFloat(a, b float64) float64 {
    if a < b {
        return a
    }
    return b
}
