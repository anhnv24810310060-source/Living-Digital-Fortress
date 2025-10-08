package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"shieldx/shared/shieldx-common/pkg/policy"
)

func main() {
	log.SetFlags(0)
	if len(os.Args) < 2 {
		usage()
		os.Exit(1)
	}
	cmd := os.Args[1]
	switch cmd {
	case "bundle":
		fs := flag.NewFlagSet("bundle", flag.ExitOnError)
		dir := fs.String("dir", "policies/demo", "policy directory containing manifest.json and .rego files")
		out := fs.String("out", "dist/policy-bundle.zip", "output zip path")
		_ = fs.Parse(os.Args[2:])
		digest, err := policy.BuildAndWrite(*dir, *out)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("bundle:", *out)
		fmt.Println("digest:", digest)
	case "sign":
		fs := flag.NewFlagSet("sign", flag.ExitOnError)
		dir := fs.String("dir", "policies/demo", "policy directory")
		out := fs.String("sig", "dist/policy-bundle.sig", "signature output path")
		_ = fs.Parse(os.Args[2:])
		b, err := policy.LoadFromDir(*dir)
		if err != nil {
			log.Fatal(err)
		}
		digest, err := b.Hash()
		if err != nil {
			log.Fatal(err)
		}
		sig, err := policy.SignDigest(policy.NoopSigner{}, digest)
		if err != nil {
			log.Fatal(err)
		}
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
			log.Fatal(err)
		}
		if err := os.WriteFile(*out, sig, 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Println("signed:", *out)
		fmt.Println("digest:", digest)
	case "verify":
		fs := flag.NewFlagSet("verify", flag.ExitOnError)
		dir := fs.String("dir", "policies/demo", "policy directory")
		sigPath := fs.String("sig", "dist/policy-bundle.sig", "signature file path")
		_ = fs.Parse(os.Args[2:])
		b, err := policy.LoadFromDir(*dir)
		if err != nil {
			log.Fatal(err)
		}
		digest, err := b.Hash()
		if err != nil {
			log.Fatal(err)
		}
		sig, err := os.ReadFile(*sigPath)
		if err != nil {
			log.Fatal(err)
		}
		if err := policy.VerifyDigest(policy.NoopVerifier{}, digest, sig); err != nil {
			log.Fatal(err)
		}
		fmt.Println("verify: OK")
		fmt.Println("digest:", digest)
	default:
		usage()
		os.Exit(1)
	}
}

func usage() {
	fmt.Println("policyctl <bundle|sign|verify> [flags]")
}
