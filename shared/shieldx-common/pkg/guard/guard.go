package guard

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/hex"
    "net/http"
    "strconv"
    "strings"
    "time"
)

// Compute returns HMAC-SHA256(secret, minuteBucket||scope) in hex.
func Compute(secret string, minute int64, scope string) string {
    h := hmac.New(sha256.New, []byte(secret))
    h.Write([]byte(strconv.FormatInt(minute, 10)))
    h.Write([]byte("|"))
    h.Write([]byte(scope))
    return hex.EncodeToString(h.Sum(nil))
}

// VerifyHeader checks X-Admission (or header override) against current/previous minute bucket.
func VerifyHeader(r *http.Request, secret string, headerName string, scope string) bool {
    if secret == "" { return true }
    if headerName == "" { headerName = "X-Admission" }
    token := r.Header.Get(headerName)
    if token == "" {
        // allow query fallback
        token = r.URL.Query().Get(strings.ToLower(headerName))
    }
    if token == "" { return false }
    now := time.Now().Unix() / 60
    if token == Compute(secret, now, scope) { return true }
    if token == Compute(secret, now-1, scope) { return true }
    return false
}



