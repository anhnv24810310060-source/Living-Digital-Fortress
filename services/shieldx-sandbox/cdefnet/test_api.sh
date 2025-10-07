#!/bin/bash

# CDefNet API Test Script

BASE_URL="http://localhost:8090"
TOKEN="demo_token_12345678901234567890123456789012"

echo "=== CDefNet API Testing ==="

# Test 1: Health Check
echo "1. Testing health endpoint..."
curl -s "$BASE_URL/health" | jq .
echo ""

# Test 2: Submit IOC - Hash
echo "2. Submitting hash IOC..."
curl -s -X POST "$BASE_URL/v1/submit-ioc" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "ioc_type": "hash",
    "value": "5d41402abc4b2a76b9719d911017c592",
    "confidence": 0.9,
    "ttl": 3600
  }' | jq .
echo ""

# Test 3: Submit IOC - Domain
echo "3. Submitting domain IOC..."
curl -s -X POST "$BASE_URL/v1/submit-ioc" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "ioc_type": "domain",
    "value": "evil.com",
    "confidence": 0.8,
    "ttl": 7200
  }' | jq .
echo ""

# Test 4: Submit IOC - IP with PII (should be scrubbed)
echo "4. Submitting IP IOC with PII..."
curl -s -X POST "$BASE_URL/v1/submit-ioc" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "ioc_type": "ip",
    "value": "192.168.1.100 contact admin@evil.com",
    "confidence": 0.7,
    "ttl": 1800
  }' | jq .
echo ""

# Test 5: Query IOC - Hash
echo "5. Querying hash IOC..."
curl -s "$BASE_URL/v1/query-ioc?type=hash&value=5d41402abc4b2a76b9719d911017c592" | jq .
echo ""

# Test 6: Query IOC - Domain
echo "6. Querying domain IOC..."
curl -s "$BASE_URL/v1/query-ioc?type=domain&value=evil.com" | jq .
echo ""

# Test 7: Query non-existent IOC
echo "7. Querying non-existent IOC..."
curl -s "$BASE_URL/v1/query-ioc?type=hash&value=nonexistent" | jq .
echo ""

# Test 8: Invalid IOC submission
echo "8. Testing invalid IOC submission..."
curl -s -X POST "$BASE_URL/v1/submit-ioc" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "ioc_type": "hash",
    "value": "invalid_hash",
    "confidence": 0.5,
    "ttl": 3600
  }' | jq .
echo ""

# Test 9: Unauthorized request
echo "9. Testing unauthorized request..."
curl -s -X POST "$BASE_URL/v1/submit-ioc" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "ioc_type": "hash",
    "value": "5d41402abc4b2a76b9719d911017c592",
    "confidence": 0.5,
    "ttl": 3600
  }'
echo ""

# Test 10: Rate limiting (send multiple requests quickly)
echo "10. Testing rate limiting..."
for i in {1..5}; do
  curl -s -X POST "$BASE_URL/v1/submit-ioc" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
      \"tenant_id\": \"demo\",
      \"ioc_type\": \"hash\",
      \"value\": \"hash_$i\",
      \"confidence\": 0.5,
      \"ttl\": 3600
    }" &
done
wait
echo ""

# Test 11: Metrics endpoint
echo "11. Testing metrics endpoint..."
curl -s "$BASE_URL/metrics"
echo ""

echo "=== Testing Complete ==="