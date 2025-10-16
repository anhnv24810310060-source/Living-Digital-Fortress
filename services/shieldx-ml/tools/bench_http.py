#!/usr/bin/env python3
import asyncio
import aiohttp
import numpy as np
import json
import time
from statistics import mean


async def predict(session, url, batch=32, input_dim=50):
    data = np.random.randn(batch, input_dim).tolist()
    payload = {"data": data, "return_proba": False}
    async with session.post(url, json=payload) as resp:
        await resp.text()
        return resp.status


async def run_benchmark(base_url: str, model_name: str, concurrency: int = 1000, requests_per_client: int = 10):
    url = f"{base_url}/models/{model_name}/predict"
    timeout = aiohttp.ClientTimeout(total=30)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        latencies = []

        async def worker():
            for _ in range(requests_per_client):
                start = time.perf_counter_ns()
                status = await predict(session, url)
                end = time.perf_counter_ns()
                if status == 200:
                    latencies.append((end - start) / 1e6)  # ms

        tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
        t0 = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - t0

        if not latencies:
            print("No successful requests.")
            return

        latencies.sort()
        p50 = latencies[int(0.50 * len(latencies))]
        p90 = latencies[int(0.90 * len(latencies))]
        p99 = latencies[int(0.99 * len(latencies))]
        print(f"Requests: {len(latencies)} in {elapsed:.2f}s, RPS={len(latencies)/elapsed:.1f}")
        print(f"Latency ms: p50={p50:.2f}, p90={p90:.2f}, p99={p99:.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--model", default="autoencoder_demo")
    parser.add_argument("--concurrency", type=int, default=1000)
    parser.add_argument("--rpc", type=int, default=10, help="requests per client")
    args = parser.parse_args()
    asyncio.run(run_benchmark(args.base_url, args.model, args.concurrency, args.rpc))
