"""
KlavoraAI vLLM Client

Test your fine-tuned adapters with this client.
Supports streaming, function calling patterns, and batch requests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Optional

try:
    import httpx
except ImportError:
    print("httpx required: pip install httpx")
    sys.exit(1)


class KlavoraAIClient:
    """Client for interacting with KlavoraAI vLLM API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        if api_key:
            self.client.headers["Authorization"] = f"Bearer {api_key}"

    def list_models(self) -> list[dict[str, Any]]:
        """List available models."""
        response = self.client.get(f"{self.base_url}/models")
        response.raise_for_status()
        data = response.json()
        return data.get("data", [])

    def generate(
        self,
        prompt: str,
        model: str = "StMark007/klavora-contract-qwen3-4b",
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Generate completion for a prompt.

        Args:
            prompt: Input text prompt
            model: Model/adapter to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Enable streaming response

        Returns:
            API response with generated text
        """
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        response = self.client.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    def extract_contract(self, contract_text: str, model: str = "StMark007/klavora-contract-qwen3-4b") -> dict[str, Any]:
        """
        Extract structured data from a contract.

        Args:
            contract_text: Raw contract text
            model: Contract extraction adapter

        Returns:
            Parsed contract extraction JSON
        """
        prompt = (
            "Mode: contract\n"
            "Task: extract structured data into the canonical JSON schema.\n\n"
            "Return only valid JSON.\n"
            "Do not repeat the document.\n"
            "If a field is unknown, use null or [] as appropriate.\n\n"
            f"Document:\n{contract_text}"
        )

        result = self.generate(prompt=prompt, model=model, max_tokens=1024, temperature=0.1)
        content = result["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse JSON from response: {content[:200]}")

    def extract_policy(self, policy_text: str, model: str = "StMark007/klavora-policy-qwen3-4b") -> dict[str, Any]:
        """
        Extract structured data from a policy document.

        Args:
            policy_text: Raw policy text
            model: Policy extraction adapter

        Returns:
            Parsed policy extraction JSON
        """
        prompt = (
            "Mode: policy\n"
            "Task: extract structured data into the canonical JSON schema.\n\n"
            "Return only valid JSON.\n"
            "Do not repeat the document.\n"
            "If a field is unknown, use null or [] as appropriate.\n\n"
            f"Document:\n{policy_text}"
        )

        result = self.generate(prompt=prompt, model=model, max_tokens=1024, temperature=0.1)
        content = result["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise ValueError(f"Could not parse JSON from response: {content[:200]}")

    def benchmark(
        self,
        prompts: list[str],
        model: str = "StMark007/klavora-contract-qwen3-4b",
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """
        Run simple benchmark on a set of prompts.

        Returns latency and throughput metrics.
        """
        latencies = []
        tokens_generated = 0

        for prompt in prompts:
            start = time.perf_counter()
            result = self.generate(prompt=prompt, model=model, max_tokens=max_tokens)
            latency = time.perf_counter() - start

            usage = result.get("usage", {})
            tokens = usage.get("completion_tokens", 0)

            latencies.append(latency)
            tokens_generated += tokens

        total_time = sum(latencies)
        avg_latency = total_time / len(latencies) if latencies else 0

        return {
            "num_requests": len(prompts),
            "total_time_seconds": round(total_time, 3),
            "avg_latency_seconds": round(avg_latency, 3),
            "tokens_generated": tokens_generated,
            "tokens_per_second": round(tokens_generated / total_time, 2) if total_time > 0 else 0,
            "requests_per_second": round(len(prompts) / total_time, 2) if total_time > 0 else 0,
        }

    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()


def demo_contract_extraction(client: KlavoraAIClient) -> None:
    """Demo: Extract data from a sample contract."""
    sample_contract = """
    NON-DISCLOSURE AGREEMENT

    This Non-Disclosure Agreement ("Agreement") is entered into as of January 15, 2024,
    between Acme Corporation ("Disclosing Party") and Beta Inc ("Receiving Party").

    The Disclosing Party agrees to share certain confidential information with the Receiving Party
    for the purpose of evaluating a potential business partnership.

    The Receiving Party agrees to:
    - Keep all information confidential
    - Not disclose to any third party
    - Use the information only for the stated purpose

    This Agreement shall remain in effect for two (2) years from the Effective Date.
    """

    print("\n--- Contract Extraction Demo ---")
    result = client.extract_contract(sample_contract)
    print(json.dumps(result, indent=2))


def demo_policy_extraction(client: KlavoraAIClient) -> None:
    """Demo: Extract data from a sample policy."""
    sample_policy = """
    PRIVACY POLICY

    Last Updated: March 1, 2024

    This Privacy Policy describes how TechCorp Inc collects, uses, and shares
    information about users of our website and mobile application.

    Information We Collect:
    - Personal information you provide directly
    - Device and browser information
    - Cookies and tracking data

    We may share information with:
    - Third-party advertising partners
    - Service providers
    - Legal authorities when required

    Users have the right to:
    - Access their personal data
    - Request deletion of their data
    - Opt out of targeted advertising
    """

    print("\n--- Policy Extraction Demo ---")
    result = client.extract_policy(sample_policy)
    print(json.dumps(result, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="KlavoraAI vLLM Client")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--demo", choices=["contract", "policy", "benchmark", "list"], help="Run demo")
    parser.add_argument("--model", default="StMark007/klavora-contract-qwen3-4b", help="Model to use")
    parser.add_argument("--benchmark-prompts", type=int, default=10, help="Number of benchmark prompts")
    args = parser.parse_args()

    client = KlavoraAIClient(base_url=args.url)

    if args.demo == "list":
        print("\n--- Available Models ---")
        for model in client.list_models():
            print(f"  - {model['id']}")
        return 0

    if args.demo == "contract":
        demo_contract_extraction(client)
        return 0

    if args.demo == "policy":
        demo_policy_extraction(client)
        return 0

    if args.demo == "benchmark":
        print(f"\n--- Benchmarking {args.model} ---")
        prompts = [
            "Extract the key terms from this NDA between Company A and Company B.",
            "What are the payment terms in this service agreement?",
            "Identify all termination clauses in this contract.",
        ] * (args.benchmark_prompts // 3 + 1)
        prompts = prompts[:args.benchmark_prompts]

        result = client.benchmark(prompts, model=args.model)
        print(json.dumps(result, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
