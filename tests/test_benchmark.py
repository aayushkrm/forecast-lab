import contextlib
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from scripts import benchmark


class ForecastHandler(BaseHTTPRequestHandler):
    calls = 0
    def do_POST(self):
        type(self).calls += 1
        length = int(self.headers["Content-Length"])
        data = json.loads(self.rfile.read(length))
        assert len(data["history"]) == 168 and data["next_index"] == 168
        response = b'{"prediction": 1.0, "next_index": 168, "model_version": 1}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, *args):
        pass


class BenchmarkTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(benchmark.percentile([30, 10, 20], 0.95), 29)
        self.assertEqual(benchmark.percentile([42], 0.5), 42)

    def test_http_warmup_excluded_and_output_is_machine_readable(self):
        ForecastHandler.calls = 0
        server = ThreadingHTTPServer(("127.0.0.1", 0), ForecastHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = benchmark.main(["--url", f"http://127.0.0.1:{server.server_port}/forecast", "--requests", "3", "--warmup", "2"])
            result = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(ForecastHandler.calls, 5)
            self.assertEqual(result["requests"], 3)
            self.assertEqual(result["warmup"], 2)
            self.assertGreater(result["latency_ms"]["p95"], 0)
            self.assertGreater(result["requests_per_second"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_request_failure_returns_nonzero(self):
        with patch.object(benchmark, "request_forecast", side_effect=OSError("offline")), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(benchmark.main(["--requests", "1", "--warmup", "0"]), 1)


if __name__ == "__main__":
    unittest.main()
