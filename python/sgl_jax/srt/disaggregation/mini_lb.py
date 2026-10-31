from __future__ import annotations

import asyncio
import logging
import os
import random
from http import HTTPStatus
from itertools import chain

logger = logging.getLogger(__name__)

AIOHTTP_STREAM_READ_CHUNK_SIZE = 1024 * 64


async def _iter_stream_lines(content):
    # Cumulative logprobs can exceed aiohttp's readline high-water mark.
    # Read bounded transport chunks and reconstruct complete SSE lines ourselves.
    pending = bytearray()
    async for chunk in content.iter_chunked(AIOHTTP_STREAM_READ_CHUNK_SIZE):
        pending.extend(chunk)
        start = 0
        while (end := pending.find(b"\n", start)) >= 0:
            yield bytes(pending[start : end + 1])
            start = end + 1
        if start:
            del pending[:start]
    if pending:
        yield bytes(pending)


def _get_dp_size(server_info: dict) -> int:
    dp_size = server_info.get("dp_size")
    if dp_size is not None:
        return int(dp_size)
    return max(1, len(server_info.get("internal_states") or []))


def _prepend_input_logprobs(prefill_meta: dict, response: dict) -> None:
    # Output-only logprob requests legitimately omit prompt logprobs on P.
    if "input_token_logprobs" not in prefill_meta:
        return
    decode_meta = response.setdefault("meta_info", {})
    decode_meta["input_token_logprobs"] = (prefill_meta["input_token_logprobs"] or []) + (
        decode_meta.get("input_token_logprobs") or []
    )


class MiniLoadBalancer:
    def __init__(self, router_args):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.test_external_dp_routing = router_args.test_external_dp_routing
        self.prefill_bootstrap_host = router_args.prefill_bootstrap_host
        self.prefill_dp_size = None
        self.decode_dp_size = None
        self.max_concurrent_requests = getattr(router_args, "max_concurrent_requests", None)
        self._sticky_dp_map: dict[object, int] = {}
        self._tree_dp_map: dict[str, int] = {}
        self._rr_counter: int = 0
        self._p_active_reqs: list[int] = []
        self._p_live_reqs: list[int] = []
        self._p_active_chars: list[int] = []
        self._p_conv_count: list[int] = []
        self._d_active_reqs: list[int] = []
        self._d_decoding_reqs: list[int] = []
        self._d_active_chars: list[int] = []
        self._d_total_reqs: list[int] = []

    def _validate_router_args(self, router_args) -> None:
        if getattr(router_args, "policy", "random") != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not getattr(router_args, "pd_disaggregation", False):
            raise ValueError("MiniLB only supports PD disaggregation mode")
        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError("MiniLB requires at least one prefill and one decode server")

    def start(self) -> None:
        import uvicorn

        global lb, _admission_sem
        lb = self
        if self.max_concurrent_requests:
            _admission_sem = asyncio.Semaphore(self.max_concurrent_requests)
        uvicorn.run(app, host=self.host, port=self.port)

    async def _ensure_dp_sizes(self) -> None:
        if self.prefill_dp_size is not None:
            return

        import aiohttp

        async with aiohttp.ClientSession() as session:
            prefill_info = await fetch_backend_json(
                session,
                self.prefill_urls[0],
                ("get_server_info", "server_info"),
            )
            decode_info = await fetch_backend_json(
                session,
                self.decode_urls[0],
                ("get_server_info", "server_info"),
            )
        self.prefill_dp_size = _get_dp_size(prefill_info)
        self.decode_dp_size = _get_dp_size(decode_info)
        logger.info(
            "[MiniLB] DP sizes: prefill=%s, decode=%s",
            self.prefill_dp_size,
            self.decode_dp_size,
        )

    def _fork_dp_requests(self, request: dict):
        if self.prefill_dp_size != self.decode_dp_size:
            raise RuntimeError(
                "PD Raiden requires homogeneous DP topology: "
                f"prefill dp_size={self.prefill_dp_size}, "
                f"decode dp_size={self.decode_dp_size}"
            )
        p_rank = random.randint(0, self.prefill_dp_size - 1)
        d_rank = random.randint(0, self.decode_dp_size - 1)

        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req.pop("_x_correlation_id", None)
        decode_req.pop("_x_correlation_id", None)
        prefill_req["dp_rank"] = p_rank
        decode_req["dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank

        return prefill_req, decode_req, d_rank

    @staticmethod
    def _room_rank(bootstrap_room, dp_size: int):
        if isinstance(bootstrap_room, list):
            return [int(room) % dp_size for room in bootstrap_room]
        return int(bootstrap_room) % dp_size

    def _ensure_load_arrays(self, p_size: int, d_size: int) -> None:
        if len(self._p_active_reqs) != p_size:
            self._p_active_reqs = [0] * p_size
            self._p_live_reqs = [0] * p_size
            self._p_active_chars = [0] * p_size
            self._p_conv_count = [0] * p_size
        if len(self._d_active_reqs) != d_size:
            self._d_active_reqs = [0] * d_size
            self._d_decoding_reqs = [0] * d_size
            self._d_active_chars = [0] * d_size
            self._d_total_reqs = [0] * d_size

    @staticmethod
    def _estimate_request_chars(request: dict) -> int:
        msgs = request.get("messages")
        total = 0
        if isinstance(msgs, list):
            for m in msgs:
                if isinstance(m, dict):
                    total += len(str(m.get("content", "")))
        elif isinstance(request.get("text"), str):
            total = len(request["text"])
        elif isinstance(request.get("prompt"), str):
            total = len(request["prompt"])
        return max(total, 256)

    def _select_sticky_p_and_loaded_d(self, request: dict, p_size: int, d_size: int) -> tuple[int, int]:
        self._ensure_load_arrays(p_size, d_size)
        msgs = request.get("messages")
        candidate_keys: list[object] = []
        x_corr_id = request.get("_x_correlation_id")
        if isinstance(x_corr_id, str) and x_corr_id:
            candidate_keys.append(("cid", x_corr_id))
        if isinstance(msgs, list) and msgs:
            sys_prefix = (
                str(msgs[0].get("content", ""))[:256]
                if isinstance(msgs[0], dict)
                else ""
            )
            first_user_content = ""
            for m in msgs:
                if isinstance(m, dict) and m.get("role") == "user":
                    first_user_content = str(m.get("content", ""))
                    break
            if len(msgs) >= 4 and isinstance(msgs[2], dict):
                candidate_keys.append(
                    (
                        "t1",
                        sys_prefix,
                        first_user_content[:160],
                        str(msgs[2].get("content", ""))[:160],
                    )
                )
            if not x_corr_id:
                if first_user_content:
                    candidate_keys.append(("u0", sys_prefix, first_user_content[:256]))
                head = msgs[:2] if len(msgs) >= 2 else msgs[:1]
                candidate_keys.append(
                    repr(
                        [
                            (m.get("role"), m.get("content"))
                            for m in head
                            if isinstance(m, dict)
                        ]
                    )
                )

        p_rank: int | None = None
        for k in candidate_keys:
            if k in self._sticky_dp_map:
                p_rank = self._sticky_dp_map[k]
                break

        if p_rank is None:
            p_rank = min(
                range(p_size),
                key=lambda r: (
                    self._p_live_reqs[r] + self._p_active_reqs[r],
                    self._p_active_reqs[r],
                    self._p_active_chars[r],
                    self._p_conv_count[r],
                    r,
                ),
            )
            self._p_conv_count[p_rank] += 1

        for k in candidate_keys:
            if len(self._sticky_dp_map) >= 16384 and k not in self._sticky_dp_map:
                self._sticky_dp_map.pop(next(iter(self._sticky_dp_map)))
            self._sticky_dp_map[k] = p_rank

        d_rank = min(
            range(d_size),
            key=lambda r: (
                self._d_active_reqs[r] + self._d_decoding_reqs[r],
                self._d_active_chars[r],
                self._d_total_reqs[r],
                r,
            ),
        )
        self._d_total_reqs[d_rank] += 1
        return p_rank, d_rank

    async def _align_dp_requests(self, request: dict):
        await self._ensure_dp_sizes()
        p_size = int(self.prefill_dp_size or 1)
        d_size = int(self.decode_dp_size or 1)
        if p_size != d_size:
            raise RuntimeError(
                "PD Raiden requires homogeneous DP topology: "
                f"prefill dp_size={p_size}, decode dp_size={d_size}"
            )

        forced_rank = os.getenv("SGLANG_JAX_PD_FORCE_DP_RANK")
        if forced_rank is not None:
            forced = int(forced_rank)
            if not 0 <= forced < p_size:
                raise ValueError(f"SGLANG_JAX_PD_FORCE_DP_RANK={forced} is outside [0, {p_size})")
            p_rank = forced
            d_rank = forced
        elif os.getenv("SGLANG_JAX_PD_STICKY_DP", "0") == "1" and not isinstance(
            request.get("bootstrap_room"), list
        ):
            p_rank, d_rank = self._select_sticky_p_and_loaded_d(request, p_size, d_size)
            room = int(request["bootstrap_room"])
            request = request.copy()
            request["bootstrap_room"] = (room // p_size) * p_size + p_rank
        else:
            p_rank = self._room_rank(request["bootstrap_room"], p_size)
            d_rank = p_rank

        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req.pop("_x_correlation_id", None)
        decode_req.pop("_x_correlation_id", None)
        prefill_req["dp_rank"] = p_rank
        decode_req["dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank
        return prefill_req, decode_req

    def select_pair(self) -> tuple[str, int | None, str]:
        pidx = random.randint(0, len(self.prefill_urls) - 1)
        didx = random.randint(0, len(self.decode_urls) - 1)
        return (
            self.prefill_urls[pidx],
            self.prefill_bootstrap_ports[pidx],
            self.decode_urls[didx],
        )

    async def generate(
        self,
        modified_request: dict,
        prefill_server: str,
        decode_server: str,
        endpoint: str,
    ):
        import aiohttp
        from fastapi.responses import ORJSONResponse

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        expected_decode_dp_rank = None
        if self.test_external_dp_routing:
            await self._ensure_dp_sizes()
            (
                prefill_req,
                decode_req,
                expected_decode_dp_rank,
            ) = self._fork_dp_requests(modified_request)
        else:
            prefill_req, decode_req = await self._align_dp_requests(modified_request)

        p_rank = prefill_req.get("dp_rank")
        d_rank = decode_req.get("dp_rank")
        track_load = (
            isinstance(p_rank, int)
            and isinstance(d_rank, int)
            and 0 <= p_rank < len(self._p_active_reqs)
            and 0 <= d_rank < len(self._d_active_reqs)
        )
        req_chars = self._estimate_request_chars(modified_request) if track_load else 0
        if track_load:
            self._p_active_reqs[p_rank] += 1
            self._p_live_reqs[p_rank] += 1
            self._p_active_chars[p_rank] += req_chars
            self._d_active_reqs[d_rank] += 1
            self._d_active_chars[d_rank] += req_chars

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                tasks = [
                    session.post(f"{prefill_server}/{endpoint}", json=prefill_req),
                    session.post(f"{decode_server}/{endpoint}", json=decode_req),
                ]
                prefill_response, decode_response = await asyncio.gather(*tasks)

                if "return_logprob" in modified_request:
                    prefill_json = await prefill_response.json()
                    ret_json = await decode_response.json()
                    _prepend_input_logprobs(prefill_json.get("meta_info", {}), ret_json)
                else:
                    ret_json = await decode_response.json()

                if expected_decode_dp_rank is not None:
                    actual = ret_json.get("meta_info", {}).get("dp_rank")
                    if actual != expected_decode_dp_rank:
                        return ORJSONResponse(
                            content={
                                "error": (
                                    f"DP rank mismatch: expected {expected_decode_dp_rank}, "
                                    f"got {actual}"
                                )
                            },
                            status_code=500,
                        )

                return ORJSONResponse(
                    content=ret_json,
                    status_code=decode_response.status,
                )
        finally:
            if track_load:
                self._p_active_reqs[p_rank] = max(0, self._p_active_reqs[p_rank] - 1)
                self._p_live_reqs[p_rank] = max(0, self._p_live_reqs[p_rank] - 1)
                self._p_active_chars[p_rank] = max(0, self._p_active_chars[p_rank] - req_chars)
                self._d_active_reqs[d_rank] = max(0, self._d_active_reqs[d_rank] - 1)
                self._d_active_chars[d_rank] = max(0, self._d_active_chars[d_rank] - req_chars)

    async def generate_stream(
        self,
        modified_request: dict,
        prefill_server: str,
        decode_server: str,
        endpoint: str = "generate",
    ):
        import aiohttp
        import orjson
        from fastapi import HTTPException
        from fastapi.responses import StreamingResponse

        if self.test_external_dp_routing:
            # Streaming cannot enforce or verify per-side DP routing the way
            # generate() does, so fail loudly instead of silently skipping it.
            raise HTTPException(
                status_code=400,
                detail="--test-external-dp-routing is not supported with streaming",
            )

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        prefill_req, decode_req = await self._align_dp_requests(modified_request)
        p_rank = prefill_req.get("dp_rank")
        d_rank = decode_req.get("dp_rank")
        track_load = (
            isinstance(p_rank, int)
            and isinstance(d_rank, int)
            and 0 <= p_rank < len(self._p_active_reqs)
            and 0 <= d_rank < len(self._d_active_reqs)
        )
        req_chars = self._estimate_request_chars(modified_request) if track_load else 0
        if track_load:
            self._p_active_reqs[p_rank] += 1
            self._p_live_reqs[p_rank] += 1
            self._p_active_chars[p_rank] += req_chars
            self._d_active_reqs[d_rank] += 1
            self._d_active_chars[d_rank] += req_chars

        async def stream_results():
            p_released = False
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as session:
                    tasks = [
                        session.post(f"{prefill_server}/{endpoint}", json=prefill_req),
                        session.post(f"{decode_server}/{endpoint}", json=decode_req),
                    ]
                    prefill_response, decode_response = await asyncio.gather(*tasks)

                    if modified_request.get("return_logprob", False):
                        prefill_meta = {}
                        async for chunk in _iter_stream_lines(prefill_response.content):
                            if chunk.startswith(b"data:") and b"[DONE]" not in chunk:
                                event_meta = orjson.loads(chunk[5:]).get("meta_info", {})
                                if "input_token_logprobs" in event_meta:
                                    # Prompt logprobs may arrive after an initial empty event.
                                    prefill_meta = event_meta
                        if track_load and not p_released:
                            self._p_active_reqs[p_rank] = max(0, self._p_active_reqs[p_rank] - 1)
                            self._p_active_chars[p_rank] = max(
                                0, self._p_active_chars[p_rank] - req_chars
                            )
                            self._d_decoding_reqs[d_rank] += 1
                            p_released = True

                        async for chunk in _iter_stream_lines(decode_response.content):
                            decoded_chunk = chunk.decode("utf-8")
                            if (
                                decoded_chunk
                                and decoded_chunk.startswith("data:")
                                and "[DONE]" not in decoded_chunk
                            ):
                                ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                                _prepend_input_logprobs(prefill_meta, ret_json)
                                yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                            else:
                                yield chunk
                    else:
                        async for chunk in decode_response.content.iter_chunked(
                            AIOHTTP_STREAM_READ_CHUNK_SIZE
                        ):
                            if track_load and not p_released:
                                self._p_active_reqs[p_rank] = max(
                                    0, self._p_active_reqs[p_rank] - 1
                                )
                                self._p_active_chars[p_rank] = max(
                                    0, self._p_active_chars[p_rank] - req_chars
                                )
                                self._d_decoding_reqs[d_rank] += 1
                                p_released = True
                            yield chunk
            finally:
                if track_load:
                    if not p_released:
                        self._p_active_reqs[p_rank] = max(0, self._p_active_reqs[p_rank] - 1)
                        self._p_active_chars[p_rank] = max(
                            0, self._p_active_chars[p_rank] - req_chars
                        )
                    else:
                        self._d_decoding_reqs[d_rank] = max(0, self._d_decoding_reqs[d_rank] - 1)
                    self._p_live_reqs[p_rank] = max(0, self._p_live_reqs[p_rank] - 1)
                    self._d_active_reqs[d_rank] = max(0, self._d_active_reqs[d_rank] - 1)
                    self._d_active_chars[d_rank] = max(0, self._d_active_chars[d_rank] - req_chars)

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import ORJSONResponse, Response, StreamingResponse

    from sgl_jax.srt.disaggregation.mini_lb_helpers import (
        get_parallel_sample_num,
        inject_bootstrap_fields,
    )

    app = FastAPI()
    lb: MiniLoadBalancer | None = None
    _admission_sem: asyncio.Semaphore | None = None

    async def fetch_backend_json(
        session,
        server_url: str,
        endpoint_candidates: tuple[str, ...],
    ) -> dict:
        last_status = None
        last_error_text = ""
        for endpoint in endpoint_candidates:
            async with session.get(f"{server_url}/{endpoint}") as response:
                if response.status == 200:
                    return await response.json()
                last_status = response.status
                last_error_text = await response.text()
        raise HTTPException(
            status_code=HTTPStatus.BAD_GATEWAY,
            detail=(
                f"Failed to get {endpoint_candidates[0]} from {server_url}. "
                f"Last status: {last_status}, Response: {last_error_text}"
            ),
        )

    @app.get("/health")
    async def health_check():
        return Response(status_code=200)

    @app.get("/health_generate")
    async def health_generate():
        import aiohttp

        async with aiohttp.ClientSession() as session:
            tasks = []
            for server in chain(lb.prefill_urls, lb.decode_urls):
                tasks.append(session.get(f"{server}/health_generate"))
            for response in asyncio.as_completed(tasks):
                await response
        return Response(status_code=200)

    @app.post("/flush_cache")
    async def flush_cache():
        import aiohttp

        if lb is not None:
            lb._sticky_dp_map.clear()
            lb._tree_dp_map.clear()
            lb._rr_counter = 0
            lb._p_active_reqs.clear()
            lb._p_live_reqs.clear()
            lb._p_active_chars.clear()
            lb._p_conv_count.clear()
            lb._d_active_reqs.clear()
            lb._d_decoding_reqs.clear()
            lb._d_active_chars.clear()
            lb._d_total_reqs.clear()
        # A backend refuses while it still holds requests; pass that on so callers retry.
        failures = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            for server in chain(lb.prefill_urls, lb.decode_urls):
                tasks.append(session.post(f"{server}/flush_cache"))
            for response in asyncio.as_completed(tasks):
                resp = await response
                if resp.status != 200:
                    failures.append(f"{resp.url}: HTTP {resp.status} {(await resp.text())[:300]}")
        if failures:
            return Response(content="\n".join(failures), status_code=HTTPStatus.BAD_REQUEST)
        return Response(content="Cache flushed on every prefill and decode server.\n", status_code=200)

    @app.get("/server_info")
    @app.get("/get_server_info")
    async def get_server_info():
        import aiohttp

        prefill_infos = []
        decode_infos = []
        all_internal_states = []

        async with aiohttp.ClientSession() as session:
            for server in lb.prefill_urls:
                prefill_infos.append(
                    await fetch_backend_json(
                        session,
                        server,
                        ("get_server_info", "server_info"),
                    )
                )
            for server in lb.decode_urls:
                info_json = await fetch_backend_json(
                    session,
                    server,
                    ("get_server_info", "server_info"),
                )
                decode_infos.append(info_json)
                if "internal_states" in info_json:
                    all_internal_states.extend(info_json["internal_states"])

        return {
            "internal_states": (
                all_internal_states
                if all_internal_states
                else [{"last_gen_throughput": 0.0, "avg_spec_accept_length": None}]
            ),
            "prefill": prefill_infos,
            "decode": decode_infos,
        }

    async def _get_model_info_impl():
        import aiohttp

        if not lb or not lb.prefill_urls:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="There is no server registered",
            )

        target_server_url = lb.prefill_urls[0]
        async with aiohttp.ClientSession() as session:
            return ORJSONResponse(
                content=await fetch_backend_json(
                    session,
                    target_server_url,
                    ("get_model_info", "model_info"),
                )
            )

    @app.get("/model_info")
    async def model_info():
        return await _get_model_info_impl()

    @app.get("/get_model_info")
    async def get_model_info():
        return await _get_model_info_impl()

    async def _do_forward(request_data: dict, endpoint_name: str):
        if get_parallel_sample_num(request_data) > 1:
            raise HTTPException(
                status_code=400,
                detail="PD mini_lb does not support parallel sampling (n > 1)",
            )
        prefill_server, bootstrap_port, decode_server = lb.select_pair()
        modified_request = inject_bootstrap_fields(
            request_data,
            prefill_server=prefill_server,
            bootstrap_port=bootstrap_port,
            bootstrap_host_override=lb.prefill_bootstrap_host,
        )
        if request_data.get("stream", False):
            return await lb.generate_stream(
                modified_request,
                prefill_server,
                decode_server,
                endpoint=endpoint_name,
            )
        return await lb.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )

    async def _forward_to_backend(request_data: dict, endpoint_name: str):
        if _admission_sem is None:
            return await _do_forward(request_data, endpoint_name)

        # Pending admission: hold the permit while the request runs. Excess
        # requests await the semaphore (held pending at the proxy) and are
        # never rejected or aborted.
        await _admission_sem.acquire()
        released = False
        try:
            resp = await _do_forward(request_data, endpoint_name)
            if isinstance(resp, StreamingResponse):
                # Streaming returns immediately; the real work happens while the
                # body is drained. Transfer permit ownership to the iterator so
                # it is released only after the stream fully completes.
                original_iter = resp.body_iterator

                async def _release_after_stream():
                    try:
                        async for chunk in original_iter:
                            yield chunk
                    finally:
                        _admission_sem.release()

                resp.body_iterator = _release_after_stream()
                released = True
            return resp
        finally:
            if not released:
                _admission_sem.release()

    def _attach_correlation_id(request_data: dict, raw_request: Request | None) -> dict:
        if raw_request is not None:
            cid = raw_request.headers.get("x-correlation-id") or raw_request.headers.get(
                "x-session-id"
            )
            if cid and "_x_correlation_id" not in request_data:
                return {**request_data, "_x_correlation_id": cid}
        return request_data

    @app.post("/generate")
    async def handle_generate_request(request_data: dict, raw_request: Request = None):
        return await _forward_to_backend(
            _attach_correlation_id(request_data, raw_request), "generate"
        )

    @app.post("/v1/chat/completions")
    async def handle_chat_completion_request(request_data: dict, raw_request: Request = None):
        return await _forward_to_backend(
            _attach_correlation_id(request_data, raw_request), "v1/chat/completions"
        )

    @app.post("/v1/completions")
    async def handle_completion_request(request_data: dict, raw_request: Request = None):
        return await _forward_to_backend(
            _attach_correlation_id(request_data, raw_request), "v1/completions"
        )

    @app.get("/v1/models")
    async def get_models():
        import aiohttp

        prefill_server = lb.prefill_urls[0]
        async with aiohttp.ClientSession() as session:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())

except ModuleNotFoundError as exc:  # pragma: no cover
    logger.warning("mini_lb web deps unavailable: %s", exc)
    app = None
    lb = None
