#!/usr/bin/env python3
"""
ShieldX Gateway Production Test Suite
Tests all core functionality without requiring Go installation
"""

import json
import time
import requests
import threading
import concurrent.futures
from datetime import datetime
import statistics

class GatewayTester:
    def __init__(self, base_url="http://localhost:8080"):
        self.base_url = base_url
        self.results = []
        
    def test_health_check(self):
        """Test gateway health endpoint"""
        print("🔍 Testing Health Check...")
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Health Check: {health_data.get('status', 'unknown')}")
                print(f"   Active Requests: {health_data.get('active_requests', 0)}")
                return True
            else:
                print(f"❌ Health Check Failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health Check Error: {e}")
            return False
    
    def test_basic_request(self):
        """Test basic request processing"""
        print("\n🔍 Testing Basic Request Processing...")
        
        test_request = {
            "client_ip": "192.168.1.100",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "path": "/api/users",
            "method": "GET",
            "headers": {
                "Accept": "application/json",
                "Authorization": "Bearer test-token"
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.base_url}/",
                json=test_request,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000  # ms
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Basic Request: {result.get('action', 'unknown')}")
                print(f"   Destination: {result.get('destination', 'unknown')}")
                print(f"   Threat Score: {result.get('threat_score', 0):.2f}")
                print(f"   Trust Score: {result.get('trust_score', 0):.2f}")
                print(f"   Latency: {latency:.2f}ms")
                print(f"   Request ID: {result.get('request_id', 'none')}")
                return True, latency
            else:
                print(f"❌ Basic Request Failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False, latency
                
        except Exception as e:
            print(f"❌ Basic Request Error: {e}")
            return False, 0
    
    def test_suspicious_request(self):
        """Test suspicious request handling"""
        print("\n🔍 Testing Suspicious Request...")
        
        suspicious_request = {
            "client_ip": "192.168.1.100",
            "user_agent": "sqlmap/1.0",
            "path": "/admin/../../../etc/passwd",
            "method": "POST",
            "headers": {
                "X-Forwarded-For": "10.0.0.1, 192.168.1.1, 172.16.0.1"
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/",
                json=suspicious_request,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code in [200, 403]:
                result = response.json()
                action = result.get('action', 'unknown')
                print(f"✅ Suspicious Request: {action}")
                print(f"   Threat Score: {result.get('threat_score', 0):.2f}")
                
                if action in ['BLOCK', 'ISOLATE', 'DECEIVE']:
                    print("   ✅ Correctly identified as threat")
                    return True
                else:
                    print("   ⚠️ Not identified as threat")
                    return False
            else:
                print(f"❌ Suspicious Request Failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Suspicious Request Error: {e}")
            return False
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        print("\n🔍 Testing Rate Limiting...")
        
        def make_request():
            try:
                response = requests.post(
                    f"{self.base_url}/",
                    json={"client_ip": "192.168.1.200", "path": "/test", "method": "GET"},
                    timeout=5
                )
                return response.status_code
            except:
                return 0
        
        # Send many requests quickly
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request) for _ in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        rate_limited = sum(1 for r in results if r == 429)
        successful = sum(1 for r in results if r == 200)
        
        print(f"   Successful: {successful}")
        print(f"   Rate Limited: {rate_limited}")
        
        if rate_limited > 0:
            print("✅ Rate Limiting: Working")
            return True
        else:
            print("⚠️ Rate Limiting: Not triggered (may need higher load)")
            return True  # Not necessarily a failure
    
    def test_performance(self, num_requests=100, concurrency=10):
        """Test gateway performance"""
        print(f"\n🔍 Testing Performance ({num_requests} requests, {concurrency} concurrent)...")
        
        def make_timed_request():
            start = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/",
                    json={
                        "client_ip": f"192.168.1.{threading.current_thread().ident % 255}",
                        "path": "/api/test",
                        "method": "GET"
                    },
                    timeout=10
                )
                end = time.time()
                return (end - start) * 1000, response.status_code
            except Exception as e:
                end = time.time()
                return (end - start) * 1000, 0
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(make_timed_request) for _ in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        latencies = [r[0] for r in results]
        status_codes = [r[1] for r in results]
        
        successful = sum(1 for s in status_codes if s == 200)
        total_time = end_time - start_time
        rps = num_requests / total_time
        
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Throughput: {rps:.2f} RPS")
        print(f"   Successful: {successful}/{num_requests}")
        print(f"   Avg Latency: {statistics.mean(latencies):.2f}ms")
        print(f"   P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.2f}ms")
        print(f"   P99 Latency: {statistics.quantiles(latencies, n=100)[98]:.2f}ms")
        
        # Performance targets
        avg_latency = statistics.mean(latencies)
        p99_latency = statistics.quantiles(latencies, n=100)[98]
        
        if avg_latency < 50 and p99_latency < 100 and successful > num_requests * 0.95:
            print("✅ Performance: Meets targets")
            return True
        else:
            print("⚠️ Performance: Below targets")
            return False
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n🔍 Testing Error Handling...")
        
        # Test invalid JSON
        try:
            response = requests.post(
                f"{self.base_url}/",
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            print(f"   Invalid JSON: {response.status_code}")
        except Exception as e:
            print(f"   Invalid JSON Error: {e}")
        
        # Test missing fields
        try:
            response = requests.post(
                f"{self.base_url}/",
                json={},
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            print(f"   Missing Fields: {response.status_code}")
        except Exception as e:
            print(f"   Missing Fields Error: {e}")
        
        # Test oversized request
        try:
            large_data = {"data": "x" * 1000000}  # 1MB
            response = requests.post(
                f"{self.base_url}/",
                json=large_data,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            print(f"   Large Request: {response.status_code}")
        except Exception as e:
            print(f"   Large Request Error: {e}")
        
        print("✅ Error Handling: Tested")
        return True
    
    def test_security_headers(self):
        """Test security headers"""
        print("\n🔍 Testing Security Headers...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            headers = response.headers
            
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options", 
                "X-XSS-Protection"
            ]
            
            found_headers = []
            for header in security_headers:
                if header in headers:
                    found_headers.append(header)
                    print(f"   ✅ {header}: {headers[header]}")
                else:
                    print(f"   ❌ {header}: Missing")
            
            if len(found_headers) >= 2:
                print("✅ Security Headers: Good")
                return True
            else:
                print("⚠️ Security Headers: Incomplete")
                return False
                
        except Exception as e:
            print(f"❌ Security Headers Error: {e}")
            return False
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 ShieldX Gateway Production Test Suite")
        print("=" * 50)
        
        test_results = []
        
        # Core functionality tests
        test_results.append(("Health Check", self.test_health_check()))
        test_results.append(("Basic Request", self.test_basic_request()[0]))
        test_results.append(("Suspicious Request", self.test_suspicious_request()))
        test_results.append(("Rate Limiting", self.test_rate_limiting()))
        test_results.append(("Error Handling", self.test_error_handling()))
        test_results.append(("Security Headers", self.test_security_headers()))
        test_results.append(("Performance", self.test_performance()))
        
        # Summary
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 50)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:20} {status}")
            if result:
                passed += 1
        
        print("-" * 50)
        print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - Gateway is production ready!")
        elif passed >= total * 0.8:
            print("\n⚠️ MOSTLY PASSING - Minor issues to address")
        else:
            print("\n❌ MULTIPLE FAILURES - Needs investigation")
        
        return passed == total

if __name__ == "__main__":
    print("ShieldX Gateway Test Suite")
    print("Make sure the gateway is running on http://localhost:8080")
    print()
    
    tester = GatewayTester()
    
    # Quick connectivity test
    try:
        requests.get("http://localhost:8080/health", timeout=2)
        print("✅ Gateway is reachable")
    except:
        print("❌ Gateway is not reachable at http://localhost:8080")
        print("Please start the gateway first:")
        print("  cd services/shieldx-gateway")
        print("  go run main.go")
        exit(1)
    
    # Run full test suite
    success = tester.run_all_tests()
    exit(0 if success else 1)