package policy

import (
	"fmt"
	"io"
	"os"
	"os/exec"
)

// CosignCLI implements Signer/Verifier using the external cosign binary.
type CosignCLI struct {
	Path   string // default: "cosign"
	KeyRef string // optional: path or KMS URI; if empty, relies on keyless env/identity
}

func (c CosignCLI) cosignPath() string {
	if c.Path != "" {
		return c.Path
	}
	return "cosign"
}

// Sign signs the provided digest string using cosign sign-blob and writes the raw signature bytes to out.
func (c CosignCLI) Sign(digest string, out io.Writer) error {
	blob, err := os.CreateTemp("", "cosign-blob-*.txt")
	if err != nil {
		return err
	}
	defer os.Remove(blob.Name())
	if _, err := blob.WriteString(digest); err != nil {
		blob.Close()
		return err
	}
	blob.Close()
	sigFile, err := os.CreateTemp("", "cosign-sig-*.sig")
	if err != nil {
		return err
	}
	sigPath := sigFile.Name()
	sigFile.Close()
	args := []string{"sign-blob", "--output-signature", sigPath, "--yes"}
	if c.KeyRef != "" {
		args = append(args, "--key", c.KeyRef)
	}
	args = append(args, blob.Name())
	cmd := exec.Command(c.cosignPath(), args...)
	if outb, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("cosign sign-blob failed: %v: %s", err, string(outb))
	}
	sig, err := os.ReadFile(sigPath)
	os.Remove(sigPath)
	if err != nil {
		return err
	}
	_, err = out.Write(sig)
	return err
}

// Verify verifies the signature for the provided digest using cosign verify-blob.
func (c CosignCLI) Verify(digest string, sig io.Reader) error {
	blob, err := os.CreateTemp("", "cosign-blob-*.txt")
	if err != nil {
		return err
	}
	defer os.Remove(blob.Name())
	if _, err := blob.WriteString(digest); err != nil {
		blob.Close()
		return err
	}
	blob.Close()
	sigFile, err := os.CreateTemp("", "cosign-sig-*.sig")
	if err != nil {
		return err
	}
	defer os.Remove(sigFile.Name())
	if _, err := io.Copy(sigFile, sig); err != nil {
		sigFile.Close()
		return err
	}
	sigFile.Close()
	args := []string{"verify-blob", "--signature", sigFile.Name()}
	if c.KeyRef != "" {
		args = append(args, "--key", c.KeyRef)
	}
	args = append(args, blob.Name())
	cmd := exec.Command(c.cosignPath(), args...)
	if outb, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("cosign verify-blob failed: %v: %s", err, string(outb))
	}
	return nil
}
