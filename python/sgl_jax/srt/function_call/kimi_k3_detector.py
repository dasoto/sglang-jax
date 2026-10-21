import json
import logging
import re

from sgl_jax.srt.entrypoints.openai.protocol import Tool
from sgl_jax.srt.function_call.base_format_detector import BaseFormatDetector
from sgl_jax.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sgl_jax.srt.reasoning_parser import (
    _K3_MESSAGE_CLOSE,
    _K3_RESPONSE_CLOSE,
    _K3_RESPONSE_OPEN,
    _K3_TOOLS_CLOSE,
    _K3_TOOLS_OPEN,
    _k3_partial_suffix_len,
    _k3_strip_partial_suffix,
    _k3_strip_response_wrappers,
)

logger = logging.getLogger(__name__)

_CALL_RE = re.compile(
    r"<\|open\|>call\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
    re.DOTALL,
)
_ARG_RE = re.compile(
    r"<\|open\|>argument\s+(?P<attrs>(?:(?!<\|sep\|>).)*?)<\|sep\|>"
    r"(?P<val>.*?)<\|close\|>argument<\|sep\|>",
    re.DOTALL,
)
_ATTR_RE = re.compile(r'(?P<k>\w+)="(?P<v>[^"]*)"')


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(attrs: str) -> dict:
    return {m["k"]: _unescape_attr(m["v"]) for m in _ATTR_RE.finditer(attrs)}


class KimiK3ToolDetector(BaseFormatDetector):
    """Detector for the Kimi-K3 XTML tool-call format."""

    def __init__(self):
        super().__init__()
        self.bot_token = _K3_TOOLS_OPEN
        self.eot_token = _K3_TOOLS_CLOSE
        self._sent_normal_idx = 0

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError("Kimi-K3 uses XTML tool-call tags")

    def build_ebnf(self, tools: list[Tool]) -> str | None:
        return None

    def _decode_call(self, attrs: str, body: str) -> dict | None:
        call_attrs = _parse_attrs(attrs)
        tool_name = call_attrs.get("tool", "")
        if not tool_name:
            return None
        arguments = {}
        for arg in _ARG_RE.finditer(body):
            arg_attrs = _parse_attrs(arg["attrs"])
            key = arg_attrs.get("key", "")
            arg_type = arg_attrs.get("type", "string")
            raw_value = arg["val"]
            if arg_type == "string":
                arguments[key] = raw_value
            else:
                try:
                    arguments[key] = json.loads(raw_value)
                except json.JSONDecodeError:
                    arguments[key] = raw_value
        return {
            "name": tool_name,
            "arguments": json.dumps(arguments, ensure_ascii=False),
        }

    def _parse_calls(self, section: str) -> list[dict]:
        return [
            call
            for m in _CALL_RE.finditer(section)
            if (call := self._decode_call(m["attrs"], m["body"])) is not None
        ]

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        open_idx = text.find(self.bot_token)
        if open_idx == -1:
            return StreamingParseResult(normal_text=_k3_strip_response_wrappers(text))
        before = _k3_strip_response_wrappers(text[:open_idx])
        try:
            section_start = open_idx + len(self.bot_token)
            close_idx = text.find(self.eot_token, section_start)
            section = (
                text[section_start:] if close_idx == -1 else text[section_start:close_idx]
            )
            calls = [
                ToolCallItem(
                    tool_index=i,
                    name=call["name"],
                    parameters=call["arguments"],
                )
                for i, call in enumerate(self._parse_calls(section))
            ]
            return StreamingParseResult(normal_text=before, calls=calls)
        except Exception as e:
            logger.error("Error in Kimi-K3 detect_and_parse: %s", e, exc_info=True)
            return StreamingParseResult(normal_text=before)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        try:
            open_idx = self._buffer.find(self.bot_token)
            if open_idx == -1:
                return StreamingParseResult(normal_text=self._emit_normal_text())

            normal_text = self._emit_normal_text(limit=open_idx)
            section = self._buffer[open_idx + len(self.bot_token) :]
            calls = []
            parsed = self._parse_calls(section)
            for call in parsed[self.current_tool_id + 1 :]:
                self.current_tool_id += 1
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": call["name"],
                    "arguments": json.loads(call["arguments"]),
                }
                self.streamed_args_for_tool[self.current_tool_id] = call["arguments"]
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=call["name"],
                        parameters=call["arguments"],
                    )
                )
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error("Error in Kimi-K3 parse_streaming_increment: %s", e, exc_info=True)
            self._buffer = ""
            self._sent_normal_idx = 0
            return StreamingParseResult()

    def _emit_normal_text(self, limit: int | None = None) -> str:
        if limit is None:
            holdback = _k3_partial_suffix_len(
                self._buffer,
                [self.bot_token, _K3_RESPONSE_OPEN, _K3_RESPONSE_CLOSE, _K3_MESSAGE_CLOSE],
            )
            limit = len(self._buffer) - holdback
        if limit <= self._sent_normal_idx:
            return ""
        pending = self._buffer[self._sent_normal_idx : limit]
        for marker in (_K3_RESPONSE_OPEN, _K3_RESPONSE_CLOSE, _K3_MESSAGE_CLOSE):
            if marker in pending:
                pending = pending.replace(marker, "")
        self._sent_normal_idx = limit
        return pending
