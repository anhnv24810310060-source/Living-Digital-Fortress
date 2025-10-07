package config

import (
    "os"
)

// Get returns an environment variable or default value.
func Get(key, def string) string {
    if v := os.Getenv(key); v != "" { return v }
    return def
}
