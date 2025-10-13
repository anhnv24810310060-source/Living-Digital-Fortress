package main

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sync"
	"testing"
)

func TestRedirectExact(t *testing.T){
	os.Setenv("INGRESS_REDIRECT", "old.com->new.com")
	os.Setenv("INGRESS_REDIRECT_SCHEME", "https")
	os.Unsetenv("INGRESS_REDIRECT_ALLOW")
	redirectInitOnce = sync.Once{}
	loadRedirectConfig()
	if !redirectEnabled { t.Fatalf("redirect not enabled") }
	req := httptest.NewRequest("GET", "http://old.com/path?a=1", nil)
	if dst, ok := chooseRedirectHost(req.Host, req.URL); !ok || dst != "new.com" { t.Fatalf("chooseRedirectHost failed: %v %v", ok, dst) }
	if !allowedRedirectTarget("new.com") { t.Fatalf("allowedRedirectTarget should allow new.com") }
	loc := buildRedirectURL("new.com", req.URL)
	u, _ := url.Parse(loc)
	if u.Scheme != "https" || u.Host != "new.com" || u.Path != "/path" || u.RawQuery != "a=1" {
		t.Fatalf("bad url: %s", loc)
	}
}

func TestRedirectWildcard(t *testing.T){
	os.Setenv("INGRESS_REDIRECT", "*.tenant.io->customer.example.com")
	os.Unsetenv("INGRESS_REDIRECT_ALLOW")
	redirectInitOnce = sync.Once{}
	loadRedirectConfig()
	req := httptest.NewRequest("GET", "http://abc.tenant.io/", nil)
	dst, ok := chooseRedirectHost(req.Host, req.URL)
	if !ok || dst != "customer.example.com" { t.Fatalf("wildcard failed") }
}

func TestRedirectDefault(t *testing.T){
	os.Setenv("INGRESS_REDIRECT", "default->d.example.com")
	os.Unsetenv("INGRESS_REDIRECT_ALLOW")
	redirectInitOnce = sync.Once{}
	loadRedirectConfig()
	req := httptest.NewRequest("GET", "http://foo.bar/", nil)
	dst, ok := chooseRedirectHost(req.Host, req.URL)
	if !ok || dst != "d.example.com" { t.Fatalf("default failed") }
}

func TestRedirectAllowlist(t *testing.T){
	os.Setenv("INGRESS_REDIRECT", "old.com->evil.com")
	os.Setenv("INGRESS_REDIRECT_ALLOW", "good.com")
	redirectInitOnce = sync.Once{}
	loadRedirectConfig()
	if allowedRedirectTarget("evil.com") { t.Fatalf("evil.com should not be allowed") }
}

// no-op
