package ml

import (
	"fmt"
	"math"
	"sync"
)

// OneClassSVM implements One-Class Support Vector Machine for anomaly detection
type OneClassSVM struct {
	mu sync.RWMutex

	// Hyperparameters
	nu          float64   // Upper bound on fraction of outliers (0 < nu <= 1)
	gamma       float64   // RBF kernel parameter
	kernel      string    // Kernel type: "rbf", "linear", "poly"
	degree      int       // Degree for polynomial kernel
	tolerance   float64   // Training convergence tolerance
	maxIter     int       // Maximum training iterations
	
	// Model parameters
	supportVectors [][]float64 // Support vectors from training
	alphas         []float64   // Lagrange multipliers
	rho            float64     // Decision boundary offset
	trained        bool
	
	// Statistics
	numFeatures    int
	numSupport     int
}

// OneClassSVMConfig configures the SVM
type OneClassSVMConfig struct {
	Nu        float64 // Fraction of outliers (default: 0.1)
	Gamma     float64 // RBF gamma (default: auto = 1/n_features)
	Kernel    string  // "rbf", "linear", "poly"
	Degree    int     // For polynomial kernel
	Tolerance float64 // Convergence tolerance
	MaxIter   int     // Max iterations
}

// NewOneClassSVM creates a new One-Class SVM detector
func NewOneClassSVM(config OneClassSVMConfig) *OneClassSVM {
	// Set defaults
	if config.Nu <= 0 || config.Nu > 1 {
		config.Nu = 0.1
	}
	if config.Kernel == "" {
		config.Kernel = "rbf"
	}
	if config.Degree <= 0 {
		config.Degree = 3
	}
	if config.Tolerance <= 0 {
		config.Tolerance = 1e-3
	}
	if config.MaxIter <= 0 {
		config.MaxIter = 1000
	}

	return &OneClassSVM{
		nu:        config.Nu,
		gamma:     config.Gamma,
		kernel:    config.Kernel,
		degree:    config.Degree,
		tolerance: config.Tolerance,
		maxIter:   config.MaxIter,
	}
}

// Train trains the One-Class SVM on normal data
func (svm *OneClassSVM) Train(data [][]float64) error {
	svm.mu.Lock()
	defer svm.mu.Unlock()

	if len(data) == 0 {
		return fmt.Errorf("training data is empty")
	}

	svm.numFeatures = len(data[0])
	n := len(data)

	// Auto-set gamma if not specified
	if svm.gamma <= 0 {
		svm.gamma = 1.0 / float64(svm.numFeatures)
	}

	// Compute kernel matrix
	kernelMatrix := svm.computeKernelMatrix(data)

	// Solve dual optimization using SMO (Sequential Minimal Optimization)
	alphas, rho := svm.solveQP(kernelMatrix, n)

	// Extract support vectors (where alpha > 0)
	threshold := 1e-5
	svm.supportVectors = make([][]float64, 0)
	svm.alphas = make([]float64, 0)

	for i := 0; i < n; i++ {
		if alphas[i] > threshold {
			svm.supportVectors = append(svm.supportVectors, data[i])
			svm.alphas = append(svm.alphas, alphas[i])
		}
	}

	svm.numSupport = len(svm.supportVectors)
	svm.rho = rho
	svm.trained = true

	return nil
}

// Detect returns anomaly score for a sample (negative = normal, positive = anomaly)
func (svm *OneClassSVM) Detect(sample []float64) (float64, error) {
	svm.mu.RLock()
	defer svm.mu.RUnlock()

	if !svm.trained {
		return 0, fmt.Errorf("model not trained")
	}

	if len(sample) != svm.numFeatures {
		return 0, fmt.Errorf("sample dimension mismatch: expected %d, got %d", 
			svm.numFeatures, len(sample))
	}

	// Compute decision function: f(x) = sum(alpha_i * K(x_i, x)) - rho
	score := -svm.rho

	for i := 0; i < len(svm.supportVectors); i++ {
		k := svm.kernelFunc(svm.supportVectors[i], sample)
		score += svm.alphas[i] * k
	}

	// Negative score = normal, positive = anomaly
	return -score, nil
}

// Algorithm returns the algorithm name
func (svm *OneClassSVM) Algorithm() string {
	return "one_class_svm"
}

// Predict returns binary prediction (true = anomaly)
func (svm *OneClassSVM) Predict(sample []float64) (bool, error) {
	score, err := svm.Detect(sample)
	if err != nil {
		return false, err
	}

	return score > 0, nil
}

// GetSupportVectors returns the support vectors
func (svm *OneClassSVM) GetSupportVectors() [][]float64 {
	svm.mu.RLock()
	defer svm.mu.RUnlock()

	result := make([][]float64, len(svm.supportVectors))
	for i, sv := range svm.supportVectors {
		result[i] = make([]float64, len(sv))
		copy(result[i], sv)
	}
	return result
}

// GetNumSupport returns the number of support vectors
func (svm *OneClassSVM) GetNumSupport() int {
	svm.mu.RLock()
	defer svm.mu.RUnlock()
	return svm.numSupport
}

// kernelFunc computes kernel between two vectors
func (svm *OneClassSVM) kernelFunc(x1, x2 []float64) float64 {
	switch svm.kernel {
	case "rbf":
		return svm.rbfKernel(x1, x2)
	case "linear":
		return svm.linearKernel(x1, x2)
	case "poly":
		return svm.polyKernel(x1, x2)
	default:
		return svm.rbfKernel(x1, x2)
	}
}

// rbfKernel computes RBF (Gaussian) kernel
func (svm *OneClassSVM) rbfKernel(x1, x2 []float64) float64 {
	sumSq := 0.0
	for i := 0; i < len(x1); i++ {
		diff := x1[i] - x2[i]
		sumSq += diff * diff
	}
	return math.Exp(-svm.gamma * sumSq)
}

// linearKernel computes linear kernel (dot product)
func (svm *OneClassSVM) linearKernel(x1, x2 []float64) float64 {
	sum := 0.0
	for i := 0; i < len(x1); i++ {
		sum += x1[i] * x2[i]
	}
	return sum
}

// polyKernel computes polynomial kernel
func (svm *OneClassSVM) polyKernel(x1, x2 []float64) float64 {
	dot := svm.linearKernel(x1, x2)
	return math.Pow(dot+1.0, float64(svm.degree))
}

// computeKernelMatrix computes the kernel matrix for training data
func (svm *OneClassSVM) computeKernelMatrix(data [][]float64) [][]float64 {
	n := len(data)
	K := make([][]float64, n)
	for i := 0; i < n; i++ {
		K[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			K[i][j] = svm.kernelFunc(data[i], data[j])
		}
	}
	return K
}

// solveQP solves the quadratic programming problem using simplified SMO
func (svm *OneClassSVM) solveQP(K [][]float64, n int) ([]float64, float64) {
	// Initialize alphas uniformly
	alphas := make([]float64, n)
	C := 1.0 / (float64(n) * svm.nu)
	
	for i := 0; i < n; i++ {
		alphas[i] = 0.5 * C
	}

	// SMO iterations with better optimization
	for iter := 0; iter < svm.maxIter; iter++ {
		alphaChanged := 0

		for i := 0; i < n; i++ {
			// Compute decision value for sample i
			fi := 0.0
			for j := 0; j < n; j++ {
				fi += alphas[j] * K[i][j]
			}

			// Compute error
			Ei := fi - 1.0

			// Check KKT conditions
			if (alphas[i] < C-svm.tolerance && Ei < -svm.tolerance) || 
			   (alphas[i] > svm.tolerance && Ei > svm.tolerance) {
				
				// Select j with maximum |Ei - Ej|
				j := svm.selectJ(i, n, alphas, K)
				
				// Compute decision value for sample j
				fj := 0.0
				for k := 0; k < n; k++ {
					fj += alphas[k] * K[j][k]
				}
				Ej := fj - 1.0

				// Save old alphas
				alphaJOld := alphas[j]

				// Compute bounds
				var L, H float64
				L = math.Max(0, alphas[i]+alphas[j]-C)
				H = math.Min(C, alphas[i]+alphas[j])

				if math.Abs(L-H) < 1e-8 {
					continue
				}

				// Compute eta
				eta := 2*K[i][j] - K[i][i] - K[j][j]
				if eta >= -1e-8 {
					continue
				}

				// Update alpha j
				alphas[j] = alphas[j] - (Ej-Ei)/eta
				alphas[j] = math.Max(L, math.Min(H, alphas[j]))

				if math.Abs(alphas[j]-alphaJOld) < 1e-5 {
					continue
				}

				// Update alpha i to maintain constraint sum(alpha) = nuN
				alphas[i] = alphas[i] + (alphaJOld - alphas[j])

				alphaChanged++
			}
		}

		// Check convergence
		if alphaChanged == 0 {
			break
		}
	}

	// Compute rho (bias) using support vectors on margin
	rho := 0.0
	numBound := 0
	
	for i := 0; i < n; i++ {
		if alphas[i] > svm.tolerance && alphas[i] < C-svm.tolerance {
			fi := 0.0
			for j := 0; j < n; j++ {
				fi += alphas[j] * K[i][j]
			}
			rho += fi
			numBound++
		}
	}
	
	if numBound > 0 {
		rho /= float64(numBound)
	} else {
		// Fallback: use all support vectors
		for i := 0; i < n; i++ {
			if alphas[i] > svm.tolerance {
				fi := 0.0
				for j := 0; j < n; j++ {
					fi += alphas[j] * K[i][j]
				}
				rho += fi
				numBound++
			}
		}
		if numBound > 0 {
			rho /= float64(numBound)
		}
	}

	return alphas, rho
}

// selectJ selects the second alpha to optimize (heuristic: max |Ei - Ej|)
func (svm *OneClassSVM) selectJ(i, n int, alphas []float64, K [][]float64) int {
	// Simple heuristic: select j that maximizes |Ei - Ej|
	bestJ := (i + 1) % n
	
	// For efficiency, just use round-robin for now
	// A better heuristic would compute all errors but that's expensive
	for bestJ == i {
		bestJ = (bestJ + 1) % n
	}
	
	return bestJ
}

// GetConfig returns the current configuration
func (svm *OneClassSVM) GetConfig() OneClassSVMConfig {
	svm.mu.RLock()
	defer svm.mu.RUnlock()

	return OneClassSVMConfig{
		Nu:        svm.nu,
		Gamma:     svm.gamma,
		Kernel:    svm.kernel,
		Degree:    svm.degree,
		Tolerance: svm.tolerance,
		MaxIter:   svm.maxIter,
	}
}

// IsTrained returns whether the model is trained
func (svm *OneClassSVM) IsTrained() bool {
	svm.mu.RLock()
	defer svm.mu.RUnlock()
	return svm.trained
}
