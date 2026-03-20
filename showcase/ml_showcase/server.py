from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
SHOWCASE_DIR = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "code"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from model_a.inference.predict_price_range_simple_input import predict_one as predict_model_a
from model_b.runtime import predict_one as predict_model_b
from model_c.runtime import load_artifacts, recommend_from_item, recommend_from_profile


class ShowcaseHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SHOWCASE_DIR), **kwargs)

    def _send_json(self, payload: dict, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send_json({}, status=200)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            return self._send_json({"status": "ok"})
        if path == "/api/model-c/samples":
            loaded = load_artifacts()
            catalog = loaded["catalog"]
            sample = catalog.head(12)[["listing_id", "brand", "category", "material", "size", "tier_primary"]]
            return self._send_json({"count": int(len(sample)), "items": sample.to_dict(orient="records")})
        return super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        content_length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
        try:
            payload = json.loads(raw or "{}")
        except json.JSONDecodeError:
            return self._send_json({"detail": "Invalid JSON body"}, status=400)

        try:
            if path == "/api/predict-price":
                result = predict_model_a(
                    brand=payload["brand"],
                    category=payload["category"],
                    material=payload["material"],
                    age_months=int(payload["age_months"]),
                    size=payload["size"],
                    condition=payload["condition"],
                    original_price=float(payload["original_price"]),
                )
                final_range = result["final_price_range"]
                min_price = float(final_range["min_price"])
                max_price = float(final_range["max_price"])
                return self._send_json(
                    {
                        "model": "ModelA",
                        "final_price_range": final_range,
                        "predicted_price_mid": round((min_price + max_price) / 2.0, 2),
                        "confidence": result.get("confidence"),
                        "model_route": result.get("model_route"),
                    }
                )

            if path == "/api/model-b/predict":
                result = predict_model_b(
                    brand=payload["brand"],
                    category=payload["category"],
                    gender=payload.get("gender") or None,
                    material=payload["material"],
                    size=payload["size"],
                    condition=payload["condition"],
                    garment_age_months=int(payload["garment_age_months"]),
                    original_price=float(payload["original_price"]),
                    provider_price=float(payload["provider_price"]),
                    current_status=payload.get("current_status") or "PENDING_REVIEW",
                    listing_created_at=payload.get("listing_created_at") or None,
                    last_approved_at=payload.get("last_approved_at") or None,
                    last_reapproved_at=payload.get("last_reapproved_at") or None,
                    as_of_date=payload.get("as_of_date") or None,
                    auto_remove_stale=bool(payload.get("auto_remove_stale", False)),
                    removal_grace_months=int(payload.get("removal_grace_months", 3)),
                )
                return self._send_json(
                    {
                        "model": "ModelB",
                        "prediction": result.get("prediction"),
                        "lifecycle": result.get("lifecycle"),
                        "age_context": result.get("age_context"),
                        "derived_features": result.get("derived_features"),
                    }
                )

            if path == "/api/model-c/recommend":
                if payload.get("seed_item_id"):
                    result = recommend_from_item(
                        seed_item_id=payload["seed_item_id"],
                        top_k=int(payload.get("top_k", 5)),
                        category_filter=payload.get("category_filter") or None,
                        max_provider_price=float(payload["max_provider_price"])
                        if payload.get("max_provider_price") is not None
                        else None,
                        exclude_same_brand=bool(payload.get("exclude_same_brand", False)),
                    )
                    return self._send_json(result)
                if payload.get("liked_item_ids"):
                    result = recommend_from_profile(
                        liked_item_ids=payload["liked_item_ids"],
                        top_k=int(payload.get("top_k", 5)),
                        category_filter=payload.get("category_filter") or None,
                        max_provider_price=float(payload["max_provider_price"])
                        if payload.get("max_provider_price") is not None
                        else None,
                        exclude_same_brand=bool(payload.get("exclude_same_brand", False)),
                    )
                    return self._send_json(result)
                return self._send_json({"detail": "Provide seed_item_id or liked_item_ids"}, status=400)

            return self._send_json({"detail": "Unknown endpoint"}, status=404)
        except Exception as exc:
            return self._send_json({"detail": str(exc)}, status=400)


def port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def showcase_already_running(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return payload.get("status") == "ok"
    except Exception:
        return False


def find_free_port(start_port: int) -> int:
    port = start_port
    while port < start_port + 100:
        if not port_in_use(port):
            return port
        port += 1
    raise RuntimeError("Could not find a free port for the ML showcase.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RentAFit ML showcase server.")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("RENTAFIT_SHOWCASE_PORT", "8090")),
        help="Port to bind locally. Defaults to 8090 or RENTAFIT_SHOWCASE_PORT.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    port = args.port

    if port_in_use(port):
        if showcase_already_running(port):
            print(f"RentAFit ML Showcase is already running at http://127.0.0.1:{port}")
            return
        fallback_port = find_free_port(port + 1)
        print(f"Port {port} is busy. Starting the showcase on http://127.0.0.1:{fallback_port} instead.")
        port = fallback_port

    server = ThreadingHTTPServer(("127.0.0.1", port), ShowcaseHandler)
    print(f"RentAFit ML Showcase running at http://127.0.0.1:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
