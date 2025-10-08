package policy

import (
	"archive/zip"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"
)

// LoadFromZipReader loads a Bundle from a zip reader (expects manifest.json at root and policy files).
func LoadFromZipReader(r *zip.Reader) (*Bundle, error) {
	var mf Manifest
	files := map[string][]byte{}
	for _, f := range r.File {
		name := filepath.ToSlash(f.Name)
		rc, err := f.Open()
		if err != nil {
			return nil, err
		}
		b, err := io.ReadAll(rc)
		rc.Close()
		if err != nil {
			return nil, err
		}
		if name == "manifest.json" {
			if err := json.Unmarshal(b, &mf); err != nil {
				return nil, fmt.Errorf("parse manifest: %w", err)
			}
		} else {
			files[name] = b
		}
	}
	if mf.Name == "" {
		return nil, fmt.Errorf("manifest.json not found")
	}
	// If manifest lists policies, keep only those paths in that order when hashing
	if len(mf.Policies) > 0 {
		filtered := map[string][]byte{}
		for _, rel := range mf.Policies {
			p := filepath.ToSlash(rel)
			if b, ok := files[p]; ok {
				filtered[p] = b
			}
		}
		files = filtered
	} else {
		// otherwise, include all .rego files in sorted order
		paths := make([]string, 0, len(files))
		for p := range files {
			if strings.HasSuffix(p, ".rego") {
				paths = append(paths, p)
			}
		}
		sort.Strings(paths)
		ordered := map[string][]byte{}
		for _, p := range paths {
			ordered[p] = files[p]
		}
		files = ordered
	}
	return &Bundle{Manifest: mf, Files: files}, nil
}
