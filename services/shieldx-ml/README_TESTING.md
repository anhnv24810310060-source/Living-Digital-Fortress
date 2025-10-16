# ShieldX ML Service - Testing & Performance

This doc explains how to validate the ML service correctness and evaluate performance targets.

## Unit & Integration Tests

- Full suite exists under `services/shieldx-ml/tests` and `services/shieldx-ml/ml-service/tests`.
- Run locally (requires Python 3.11 and dependencies):

```bash
cd services/shieldx-ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
python run_tests.py
```

- Or run quick static validation without heavy deps:

```bash
python3 validate_code.py
```

- CI runs these automatically via GitHub Actions: `.github/workflows/ml-tests.yml`.

## Load & Concurrency Tests

Targets:
- 10,000 concurrent requests
- < 100ms latency (p99)
- 99% detection rate (recall)

Tools provided:
- `tools/locustfile.py`: Locust user model for HTTP inference.
- `tools/bench_http.py`: asyncio benchmark for high-concurrency testing.

Example (asyncio benchmark):
```bash
# Start the DL service first (port 8001)
python3 ml-service/dl_service.py &
# In another terminal:
python3 tools/bench_http.py --base-url http://localhost:8001 --model autoencoder_demo --concurrency 1000 --rpc 10
```

Example (Locust):
```bash
pip install locust
locust -f tools/locustfile.py --host http://localhost:8001
```

## Production Readiness Notes

- For 10k concurrent with <100ms p99, deploy with:
  - Gunicorn + Uvicorn workers or ASGI (Quart/FastAPI) for async IO
  - Enable dynamic batching and GPU via `inference_engine.py`
  - Redis-backed cache and model warmup
  - Horizontal autoscaling (K8s HPA) and a gateway (HAProxy/NGINX) with keep-alive
  - Consider Triton Inference Server + TensorRT for GPU acceleration

- Detection 99% requires calibrated thresholds and balanced datasets; validate using `evaluate` endpoint with real distributions.

- See docs/ML_MASTER_ROADMAP.md for completed optimization and monitoring features.