class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text


class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: str | None = None,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning
        self.tool_start_token = tool_start_token

        self._buffer = ""
        self.stripped_think_start = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or text.startswith(self.think_start_token)

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # Check for tool_start_token interruption
            if self.tool_start_token is not None and self.tool_start_token in processed_text:
                tool_idx = processed_text.find(self.tool_start_token)
                reasoning_text = processed_text[:tool_idx].strip()
                normal_text = processed_text[tool_idx:]
                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0]
        normal_text = splits[1].strip()

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        tokens_to_check = [self.think_start_token, self.think_end_token]
        if self.tool_start_token:
            tokens_to_check.append(self.tool_start_token)
        if any(
            token.startswith(current_text) and token != current_text for token in tokens_to_check
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text.rstrip()
            )

        # Continue with reasoning content
        if self._in_reasoning:
            # Check for tool_start_token interruption
            if self.tool_start_token and self.tool_start_token in current_text:
                tool_idx = current_text.find(self.tool_start_token)
                reasoning_text = current_text[:tool_idx]
                normal_text = current_text[tool_idx:]
                self._buffer = ""
                self._in_reasoning = False
                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=new_text)

        return StreamingParseResult()


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    Detector for DeepSeek-R1 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )
        # https://github.com/sgl-project/sglang/pull/3202#discussion_r1950153599


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True):
        # Qwen3 chat_template emits the `<think>\n` opener as part of the prompt,
        # so the completion starts mid-reasoning without the start tag. The
        # `enable_thinking=False` case is gated upstream in serving_chat
        # (`_get_reasoning_from_request`), so this detector is only invoked when
        # reasoning is active.
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )


class KimiDetector(BaseReasoningFormatDetector):
    """
    Detector for Kimi Thinking model.
    Assumes reasoning format:
      ◁think▷*(.*)◁/think▷
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(self, stream_reasoning: bool = True):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
        )


class Glm45Detector(BaseReasoningFormatDetector):
    """
    Detector for GLM-4.5 / 4.6 / 4.7 models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    GLM models use `<tool_call>` as the tool start token to switch from reasoning mode to normal mode.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
        )


class Gemma4Detector(BaseReasoningFormatDetector):
    """
    Detector for Gemma 4 model.
    Assumes reasoning format:
      <|channel>thought\n*(.*)<channel|>
    """

    def __init__(self, stream_reasoning: bool = True):
        super().__init__(
            "<|channel>thought\n",
            "<channel|>",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
        )

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        if text.startswith("<|channel>thought\n"):
            text = text[len("<|channel>thought\n") :]
            self._in_reasoning = True
        elif text.startswith("thought\n"):
            text = text[len("thought\n") :]
            self._in_reasoning = True

        if not self._in_reasoning and "<channel|>" in text:
            self._in_reasoning = True

        if not self._in_reasoning:
            return StreamingParseResult(normal_text=text)

        processed_text = text.strip()

        if self.think_end_token not in processed_text:
            return StreamingParseResult(reasoning_text=processed_text)

        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0].strip()
        normal_text = splits[1].strip()

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        tokens_to_check = [self.think_start_token, "thought\n", self.think_end_token]
        if any(
            token.startswith(current_text) and token != current_text for token in tokens_to_check
        ):
            return StreamingParseResult()

        if not self.stripped_think_start:
            if current_text.startswith("<|channel>thought\n"):
                current_text = current_text[len("<|channel>thought\n") :]
                self.stripped_think_start = True
                self._in_reasoning = True
            elif current_text.startswith("thought\n"):
                current_text = current_text[len("thought\n") :]
                self.stripped_think_start = True
                self._in_reasoning = True

        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)
            reasoning_text = current_text[:end_idx]
            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]
            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text.rstrip()
            )

        if self._in_reasoning:
            if self.stream_reasoning:
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=new_text)

        return StreamingParseResult()


class Ling3Detector(Glm45Detector):
    """Preserve unfinished reasoning so a truncated answer is not returned empty."""

    def __init__(self, stream_reasoning: bool = True):
        super().__init__(stream_reasoning=stream_reasoning, force_reasoning=True)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        result = super().detect_and_parse(text)
        if result.reasoning_text and not result.normal_text and "<tool_call>" not in text:
            return StreamingParseResult(normal_text=result.reasoning_text)
        return result


_K3_THINK_OPEN = "<|open|>think<|sep|>"
_K3_THINK_CLOSE = "<|close|>think<|sep|>"
_K3_RESPONSE_OPEN = "<|open|>response<|sep|>"
_K3_RESPONSE_CLOSE = "<|close|>response<|sep|>"
_K3_TOOLS_OPEN = "<|open|>tools<|sep|>"
_K3_TOOLS_CLOSE = "<|close|>tools<|sep|>"
_K3_MESSAGE_CLOSE = "<|close|>message<|sep|>"
_K3_PARTIAL_SUFFIXES = (
    "<|open|>",
    "<|close|>",
    _K3_THINK_CLOSE.removesuffix("<|sep|>"),
    _K3_RESPONSE_OPEN.removesuffix("<|sep|>"),
    _K3_RESPONSE_CLOSE.removesuffix("<|sep|>"),
    _K3_TOOLS_OPEN.removesuffix("<|sep|>"),
    _K3_TOOLS_CLOSE.removesuffix("<|sep|>"),
    _K3_MESSAGE_CLOSE.removesuffix("<|sep|>"),
)


def _k3_partial_suffix_len(text: str, markers: list[str]) -> int:
    best = 0
    for marker in markers:
        if not marker:
            continue
        for length in range(min(len(marker) - 1, len(text)), best, -1):
            if text.endswith(marker[:length]):
                best = length
                break
    return best


def _k3_strip_partial_suffix(text: str) -> str:
    for suffix in _K3_PARTIAL_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _k3_strip_response_wrappers(text: str) -> str:
    open_idx = text.find(_K3_RESPONSE_OPEN)
    if open_idx != -1:
        close_idx = text.find(_K3_RESPONSE_CLOSE, open_idx + len(_K3_RESPONSE_OPEN))
        if close_idx != -1:
            text = text[open_idx + len(_K3_RESPONSE_OPEN) : close_idx]
        else:
            text = text[open_idx + len(_K3_RESPONSE_OPEN) :]
    else:
        text = text.replace(_K3_RESPONSE_CLOSE, "")
    text = text.replace(_K3_MESSAGE_CLOSE, "")
    return _k3_strip_partial_suffix(text)


class KimiK3Detector(BaseReasoningFormatDetector):
    """Detector for the Kimi-K3 XTML think channel (`<|open|>think<|sep|>...<|close|>think<|sep|>`)."""

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = True):
        super().__init__(
            _K3_THINK_OPEN,
            _K3_THINK_CLOSE,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token=_K3_TOOLS_OPEN,
        )
        self._reasoning_done = False
        self._tools_passthrough = False
        self._streamed_reasoning: list[str] = []
        self._discard_delayed_think_close = False

    def _clean_content(self, text: str) -> str:
        tools_idx = text.find(_K3_TOOLS_OPEN)
        if tools_idx != -1:
            return _k3_strip_response_wrappers(text[:tools_idx]) + text[tools_idx:]
        return _k3_strip_response_wrappers(text)

    def _next_channel_idx(self, text: str, start: int = 0) -> int:
        found = [
            idx
            for token in (_K3_RESPONSE_OPEN, self.tool_start_token)
            if token and (idx := text.find(token, start)) != -1
        ]
        return min(found) if found else -1

    @staticmethod
    def _skipped_think_channel(
        text: str,
        start: int = 0,
        think_close_idx: int = -1,
        next_channel_idx: int = -1,
    ) -> bool:
        response_close_idx = text.find(_K3_RESPONSE_CLOSE, start)
        return response_close_idx != -1 and all(
            boundary_idx == -1 or response_close_idx < boundary_idx
            for boundary_idx in (think_close_idx, next_channel_idx)
        )

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        in_reasoning = self._in_reasoning or self.think_start_token in text
        if not in_reasoning and self.think_end_token not in text:
            return StreamingParseResult(normal_text=self._clean_content(text))

        open_idx = text.find(self.think_start_token)
        start = open_idx + len(self.think_start_token) if open_idx != -1 else 0
        close_idx = text.find(self.think_end_token, start)
        tools_idx = text.find(self.tool_start_token, start) if self.tool_start_token else -1
        channel_idx = self._next_channel_idx(text, start)
        if self._skipped_think_channel(text, start, close_idx, channel_idx):
            return StreamingParseResult(
                normal_text=self._clean_content(text[start:].replace(self.think_end_token, ""))
            )
        if close_idx != -1 and tools_idx != -1 and tools_idx < close_idx:
            return StreamingParseResult(
                reasoning_text=_k3_strip_partial_suffix(text[start:tools_idx]),
                normal_text=self._clean_content(text[tools_idx:]),
            )
        if close_idx == -1:
            if channel_idx != -1:
                return StreamingParseResult(
                    reasoning_text=_k3_strip_partial_suffix(text[start:channel_idx]),
                    normal_text=self._clean_content(text[channel_idx:]),
                )
            return StreamingParseResult(reasoning_text=_k3_strip_partial_suffix(text[start:]))

        reasoning_text = text[start:close_idx]
        rest = text[close_idx + len(self.think_end_token) :]
        return StreamingParseResult(
            reasoning_text=reasoning_text,
            normal_text=self._clean_content(rest),
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        if not self._in_reasoning and not self._reasoning_done:
            open_idx = self._buffer.find(self.think_start_token)
            if open_idx != -1:
                self._buffer = self._buffer[open_idx + len(self.think_start_token) :]
                self._in_reasoning = True
                self.stripped_think_start = True
            elif self.think_start_token.startswith(self._buffer):
                return StreamingParseResult()
            else:
                self._reasoning_done = True

        if self._in_reasoning:
            buf = self._buffer
            if not self.stripped_think_start:
                open_idx = buf.find(self.think_start_token)
                if open_idx != -1:
                    buf = buf[open_idx + len(self.think_start_token) :]
                    self._buffer = buf
                    self.stripped_think_start = True

            close_idx = buf.find(self.think_end_token)
            tools_idx = buf.find(self.tool_start_token) if self.tool_start_token else -1
            channel_idx = self._next_channel_idx(buf)
            if self._skipped_think_channel(
                buf, think_close_idx=close_idx, next_channel_idx=channel_idx
            ):
                replay = "".join(self._streamed_reasoning)
                self._streamed_reasoning.clear()
                self._in_reasoning = False
                self._reasoning_done = True
                self._discard_delayed_think_close = True
                return StreamingParseResult(normal_text=replay + self._drain_content())

            if close_idx != -1 and not (tools_idx != -1 and tools_idx < close_idx):
                reasoning_text = buf[:close_idx]
                self._buffer = buf[close_idx + len(self.think_end_token) :]
                self._in_reasoning = False
                self._reasoning_done = True
                self._streamed_reasoning.clear()
                return StreamingParseResult(
                    reasoning_text=reasoning_text,
                    normal_text=self._drain_content(),
                )

            if channel_idx != -1:
                reasoning_text = _k3_strip_partial_suffix(buf[:channel_idx])
                self._buffer = buf[channel_idx:]
                self._in_reasoning = False
                self._reasoning_done = True
                self._streamed_reasoning.clear()
                self._tools_passthrough = bool(
                    self.tool_start_token and buf.startswith(self.tool_start_token, channel_idx)
                )
                return StreamingParseResult(
                    reasoning_text=reasoning_text,
                    normal_text=self._drain_content(),
                )

            if not self.stream_reasoning:
                return StreamingParseResult()
            markers = [
                self.think_end_token,
                self.tool_start_token or "",
                _K3_RESPONSE_OPEN,
                _K3_RESPONSE_CLOSE,
                _K3_MESSAGE_CLOSE,
            ]
            if not self.stripped_think_start:
                markers.append(self.think_start_token)
            holdback = _k3_partial_suffix_len(buf, markers)
            emit = buf[: len(buf) - holdback] if holdback else buf
            emit = _k3_strip_partial_suffix(emit)
            self._buffer = buf[len(emit) :]
            self._streamed_reasoning.append(emit)
            return StreamingParseResult(reasoning_text=emit)

        return StreamingParseResult(normal_text=self._drain_content())

    def _drain_content(self) -> str:
        buf = self._buffer
        if not buf:
            return ""
        if self._tools_passthrough:
            holdback = (
                _k3_partial_suffix_len(buf, [self.think_end_token])
                if self._discard_delayed_think_close
                else 0
            )
            emit = buf[: len(buf) - holdback] if holdback else buf
            self._buffer = buf[len(emit) :]
            if self._discard_delayed_think_close:
                emit = emit.replace(self.think_end_token, "")
            return emit

        tools_idx = buf.find(_K3_TOOLS_OPEN)
        if tools_idx != -1:
            holdback = (
                _k3_partial_suffix_len(buf, [self.think_end_token])
                if self._discard_delayed_think_close
                else 0
            )
            emit = buf[: len(buf) - holdback] if holdback else buf
            self._buffer = buf[len(emit) :]
            head = emit[:tools_idx]
            tail = emit[tools_idx:]
            for marker in (_K3_RESPONSE_OPEN, _K3_RESPONSE_CLOSE, _K3_MESSAGE_CLOSE):
                head = head.replace(marker, "")
            if self._discard_delayed_think_close:
                head = head.replace(self.think_end_token, "")
                tail = tail.replace(self.think_end_token, "")
            self._tools_passthrough = True
            return head + tail

        markers = [_K3_RESPONSE_OPEN, _K3_RESPONSE_CLOSE, _K3_MESSAGE_CLOSE, _K3_TOOLS_OPEN]
        if self._discard_delayed_think_close:
            markers.append(self.think_end_token)
        holdback = _k3_partial_suffix_len(buf, markers)
        emit = buf[: len(buf) - holdback] if holdback else buf
        self._buffer = buf[len(emit) :]
        for marker in (_K3_RESPONSE_OPEN, _K3_RESPONSE_CLOSE, _K3_MESSAGE_CLOSE):
            emit = emit.replace(marker, "")
        if self._discard_delayed_think_close:
            emit = emit.replace(self.think_end_token, "")
        return emit


class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    Args:
        model_type (str): Type of model to parse reasoning from
        stream_reasoning (bool): If False, accumulates reasoning content until complete.
            If True, streams reasoning content as it arrives.
    """

    DetectorMap: dict[str, type[BaseReasoningFormatDetector]] = {
        "deepseek-r1": DeepSeekR1Detector,
        "qwen3": Qwen3Detector,
        "mimo": Qwen3Detector,
        "kimi": KimiDetector,
        "kimi_k3": KimiK3Detector,
        "glm45": Glm45Detector,
        "gemma4": Gemma4Detector,
        "ling3": Ling3Detector,
    }

    def __init__(self, model_type: str | None = None, stream_reasoning: bool = True):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.detector = detector_class(stream_reasoning=stream_reasoning)

    def parse_non_stream(self, full_text: str) -> tuple[str, str]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(self, chunk_text: str) -> tuple[str, str]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text
