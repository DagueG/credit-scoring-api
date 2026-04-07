"""Script to simulate realistic traffic on the Credit Scoring API.

This script generates synthetic API requests to populate logs with realistic data
for monitoring and analysis. It supports data drift simulation to test how the
system behaves under shifted data distributions.

Example:
    Basic usage with default parameters:
    $ uv run python scripts/simulate_traffic.py
    
    With 1000 requests and data drift simulation:
    $ uv run python scripts/simulate_traffic.py --n-requests 1000 --drift
    
    Against a remote API:
    $ uv run python scripts/simulate_traffic.py --url https://api.example.com --n-requests 500
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm


class TrafficSimulator:
    """Simulator for generating realistic traffic on the Credit Scoring API.
    
    Attributes:
        base_url: Base URL of the API
        n_requests: Total number of requests to send
        delay: Delay between requests in seconds
        enable_drift: Whether to simulate data drift
        clients_df: DataFrame containing available client IDs and features
        high_credit_clients: Client IDs with high credit amounts (for drift)
        all_clients: All available client IDs
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        n_requests: int = 500,
        delay: float = 0.05,
        enable_drift: bool = False,
    ):
        """Initialize the traffic simulator.
        
        Args:
            base_url: Base URL of the API (default: http://localhost:8000)
            n_requests: Number of requests to send (default: 500)
            delay: Delay between requests in seconds (default: 0.05)
            enable_drift: Enable data drift simulation (default: False)
        """
        self.base_url = base_url.rstrip("/")
        self.n_requests = n_requests
        self.delay = delay
        self.enable_drift = enable_drift
        
        # Statistics
        self.successful_requests = 0
        self.failed_requests = 0
        self.latencies: list[float] = []
        self.predictions: dict[int, int] = {0: 0, 1: 0}
        self.errors: dict[str, int] = {}
        
        # Load client data
        self.clients_df: Optional[pd.DataFrame] = None
        self.high_credit_clients: list[int] = []
        self.all_clients: list[int] = []
        
    def load_client_data(self) -> bool:
        """Load reference client data from parquet file.
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            reference_file = Path("data/clients_reference.parquet")
            if not reference_file.exists():
                print(f"❌ Error: {reference_file} not found")
                return False
            
            self.clients_df = pd.read_parquet(reference_file)
            self.all_clients = self.clients_df.index.tolist()
            
            # Get top 20% of clients by credit amount for drift simulation
            if "AMT_CREDIT" in self.clients_df.columns:
                top_20_percentile = self.clients_df["AMT_CREDIT"].quantile(0.8)
                self.high_credit_clients = self.clients_df[
                    self.clients_df["AMT_CREDIT"] >= top_20_percentile
                ].index.tolist()
            
            print(f"✓ Loaded {len(self.all_clients)} clients from reference data")
            if self.enable_drift and self.high_credit_clients:
                print(
                    f"✓ Identified {len(self.high_credit_clients)} high-credit clients "
                    f"(top 20% by AMT_CREDIT)"
                )
            return True
            
        except Exception as e:
            print(f"❌ Error loading client data: {e}")
            return False
    
    def check_health(self) -> bool:
        """Check if the API is healthy.
        
        Returns:
            bool: True if API responds to health check, False otherwise
        """
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print(f"✓ Health check passed: {self.base_url} is healthy")
                    return True
                else:
                    print(f"❌ Health check failed with status {response.status_code}")
                    return False
        except httpx.ConnectError:
            print(f"❌ Cannot connect to {self.base_url}")
            return False
        except httpx.TimeoutException:
            print(f"❌ Health check timeout to {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def select_client_id(self) -> int:
        """Select a client ID for prediction.
        
        Without drift: uniform random selection from all clients
        With drift: 80% from high credit clients, 20% random
        
        Returns:
            int: Selected client ID
        """
        if not self.enable_drift:
            return int(np.random.choice(self.all_clients))
        
        # Select 80% from high credit clients, 20% random
        if np.random.random() < 0.8 and self.high_credit_clients:
            return int(np.random.choice(self.high_credit_clients))
        else:
            return int(np.random.choice(self.all_clients))
    
    def send_request(self, client: httpx.Client, client_id: int) -> Optional[dict]:
        """Send a prediction request to the API.
        
        Args:
            client: httpx.Client instance
            client_id: Client ID for prediction
            
        Returns:
            dict with response data and metadata, or None if request failed
        """
        start_time = time.time()
        try:
            response = client.post(
                f"{self.base_url}/predict",
                json={"client_id": client_id},
                timeout=10.0,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                self.successful_requests += 1
                self.latencies.append(elapsed_ms)
                prediction = data.get("prediction")
                if prediction in [0, 1]:
                    self.predictions[prediction] += 1
                return {
                    "status": "success",
                    "latency_ms": elapsed_ms,
                    "response": data,
                }
            else:
                error_type = f"HTTP_{response.status_code}"
                self.errors[error_type] = self.errors.get(error_type, 0) + 1
                self.failed_requests += 1
                return {
                    "status": "failed",
                    "error": error_type,
                    "latency_ms": elapsed_ms,
                }
                
        except httpx.TimeoutException:
            elapsed_ms = (time.time() - start_time) * 1000
            self.errors["timeout"] = self.errors.get("timeout", 0) + 1
            self.failed_requests += 1
            return {"status": "failed", "error": "timeout", "latency_ms": elapsed_ms}
            
        except httpx.ConnectError:
            elapsed_ms = (time.time() - start_time) * 1000
            self.errors["connection_error"] = self.errors.get("connection_error", 0) + 1
            self.failed_requests += 1
            return {
                "status": "failed",
                "error": "connection_error",
                "latency_ms": elapsed_ms,
            }
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            error_type = type(e).__name__
            self.errors[error_type] = self.errors.get(error_type, 0) + 1
            self.failed_requests += 1
            return {"status": "failed", "error": error_type, "latency_ms": elapsed_ms}
    
    def run(self) -> bool:
        """Run the traffic simulation.
        
        Returns:
            bool: True if simulation completed successfully, False otherwise
        """
        # Load data
        if not self.load_client_data():
            return False
        
        # Health check
        if not self.check_health():
            return False
        
        print(f"\n📊 Generating {self.n_requests} requests...")
        if self.enable_drift:
            print("   Mode: DATA DRIFT (80% high-credit, 20% random)")
        else:
            print("   Mode: UNIFORM RANDOM")
        print()
        
        # Send requests with progress bar
        with httpx.Client(timeout=10.0) as client:
            for _ in tqdm(range(self.n_requests), desc="Requests", unit="req"):
                client_id = self.select_client_id()
                self.send_request(client, client_id)
                
                if self.delay > 0:
                    time.sleep(self.delay)
        
        return True
    
    def print_summary(self) -> None:
        """Print summary statistics of the simulation."""
        print("\n" + "=" * 70)
        print("📈 SIMULATION SUMMARY")
        print("=" * 70)
        
        # Request statistics
        total_requests = self.successful_requests + self.failed_requests
        success_rate = (self.successful_requests / total_requests * 100 
                       if total_requests > 0 else 0)
        
        print(f"\n✅ Successful requests: {self.successful_requests}")
        print(f"❌ Failed requests:     {self.failed_requests}")
        print(f"📊 Success rate:        {success_rate:.1f}%")
        
        # Latency statistics
        if self.latencies:
            latencies_sorted = sorted(self.latencies)
            avg_latency = statistics.mean(self.latencies)
            p50_latency = statistics.median(self.latencies)
            p95_latency = latencies_sorted[int(len(latencies_sorted) * 0.95)]
            
            print(f"\n⏱️  Latency Statistics (ms):")
            print(f"   Average:  {avg_latency:.2f} ms")
            print(f"   P50:      {p50_latency:.2f} ms")
            print(f"   P95:      {p95_latency:.2f} ms")
        
        # Prediction distribution
        total_predictions = sum(self.predictions.values())
        if total_predictions > 0:
            pred_0_pct = (self.predictions[0] / total_predictions * 100)
            pred_1_pct = (self.predictions[1] / total_predictions * 100)
            
            print(f"\n🎯 Prediction Distribution:")
            print(f"   Class 0 (No default): {self.predictions[0]} ({pred_0_pct:.1f}%)")
            print(f"   Class 1 (Default):    {self.predictions[1]} ({pred_1_pct:.1f}%)")
        
        # Error distribution
        if self.errors:
            print(f"\n⚠️  Error Distribution:")
            for error_type, count in sorted(self.errors.items(), 
                                            key=lambda x: x[1], reverse=True):
                print(f"   {error_type}: {count}")
        
        print("\n" + "=" * 70)
        print("✓ Simulation complete! Check logs/api_logs.jsonl for detailed logs.")
        print("=" * 70 + "\n")


def main() -> int:
    """Main entry point for the traffic simulation script.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    parser = argparse.ArgumentParser(
        description="Simulate realistic traffic on the Credit Scoring API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default behavior (500 requests to localhost)
  uv run python scripts/simulate_traffic.py
  
  # With data drift simulation
  uv run python scripts/simulate_traffic.py --n-requests 1000 --drift
  
  # Against remote API with custom delay
  uv run python scripts/simulate_traffic.py --url https://api.example.com --delay 0.1
        """,
    )
    
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=500,
        help="Number of requests to send (default: 500)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Delay between requests in seconds (default: 0.05)",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Enable data drift simulation (80%% high-credit, 20%% random)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.n_requests < 1:
        print("❌ Error: --n-requests must be >= 1")
        return 1
    
    if args.delay < 0:
        print("❌ Error: --delay must be >= 0")
        return 1
    
    # Run simulation
    simulator = TrafficSimulator(
        base_url=args.url,
        n_requests=args.n_requests,
        delay=args.delay,
        enable_drift=args.drift,
    )
    
    if simulator.run():
        simulator.print_summary()
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
