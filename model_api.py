import ast
import json
import os
import re
import threading

import openai


DEFAULT_MODEL = "GPT-OSS-20B"
_thread_local = threading.local()


def _sanitize_text(value):
    if not isinstance(value, str):
        return value
    return value.encode("utf-8", errors="replace").decode("utf-8")


def create_openai_client():
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or os.getenv("BASE_URL", "").strip()
    api_key = os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("API_KEY", "").strip() or "EMPTY"

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return openai.OpenAI(**client_kwargs)


def get_default_model():
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _set_last_completion_raw(content):
    _thread_local.last_completion_raw = content


def get_last_completion_raw():
    return getattr(_thread_local, "last_completion_raw", "")


def _set_last_completion_error(error_text):
    _thread_local.last_completion_error = error_text


def get_last_completion_error():
    return getattr(_thread_local, "last_completion_error", "")


def _set_last_completion_reasoning(reasoning_text):
    _thread_local.last_completion_reasoning = reasoning_text


def get_last_completion_reasoning():
    return getattr(_thread_local, "last_completion_reasoning", "")


def _normalize_message_field(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return _sanitize_text(value)
    if isinstance(value, list):
        normalized_chunks = []
        for item in value:
            if isinstance(item, str):
                normalized_chunks.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("content") or item.get("reasoning")
                if text_value:
                    normalized_chunks.append(str(text_value))
            else:
                text_value = getattr(item, "text", None) or getattr(item, "content", None)
                if text_value:
                    normalized_chunks.append(str(text_value))
        return _sanitize_text("\n".join(chunk for chunk in normalized_chunks if chunk))
    if isinstance(value, dict):
        return _sanitize_text(json.dumps(value, ensure_ascii=False))
    return _sanitize_text(str(value))


def _serialize_response(response):
    if hasattr(response, "model_dump_json"):
        return _sanitize_text(response.model_dump_json(indent=2))
    if hasattr(response, "model_dump"):
        return _sanitize_text(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
    return _sanitize_text(str(response))


def _extract_message_content(response):
    message = response.choices[0].message
    return _normalize_message_field(getattr(message, "content", ""))


def _extract_message_reasoning(response):
    message = response.choices[0].message
    reasoning_value = getattr(message, "reasoning", None)
    if reasoning_value:
        return _normalize_message_field(reasoning_value)

    reasoning_content_value = getattr(message, "reasoning_content", None)
    if reasoning_content_value:
        return _normalize_message_field(reasoning_content_value)

    return ""


def _extract_json_object(content):
    if not content:
        raise ValueError("Model returned an empty response.")

    def strip_code_fences(text):
        stripped_text = text.strip()
        if not stripped_text.startswith("```"):
            return stripped_text

        for block in stripped_text.split("```"):
            candidate = block.strip()
            if not candidate:
                continue
            if candidate.startswith("json"):
                candidate = candidate[4:].strip()
            if candidate:
                return candidate
        return stripped_text

    def remove_trailing_commas(text):
        previous = None
        current = text
        while previous != current:
            previous = current
            current = re.sub(r",\s*([}\]])", r"\1", current)
        return current

    def escape_control_chars_in_strings(text):
        escaped_chars = []
        inside_string = False
        escape_next = False

        for char in text:
            if inside_string:
                if escape_next:
                    escaped_chars.append(char)
                    escape_next = False
                    continue
                if char == "\\":
                    escaped_chars.append(char)
                    escape_next = True
                    continue
                if char == '"':
                    escaped_chars.append(char)
                    inside_string = False
                    continue
                if char == "\n":
                    escaped_chars.append("\\n")
                    continue
                if char == "\r":
                    escaped_chars.append("\\r")
                    continue
                if char == "\t":
                    escaped_chars.append("\\t")
                    continue
            else:
                if char == '"':
                    inside_string = True
            escaped_chars.append(char)

        return "".join(escaped_chars)

    def extract_balanced_json_substring(text):
        start_idx = -1
        stack = []
        inside_string = False
        escape_next = False

        for idx, char in enumerate(text):
            if start_idx == -1:
                if char in "{[":
                    start_idx = idx
                    stack.append("}" if char == "{" else "]")
                continue

            if inside_string:
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    inside_string = False
                continue

            if char == '"':
                inside_string = True
            elif char in "{[":
                stack.append("}" if char == "{" else "]")
            elif char in "}]":
                if not stack or char != stack[-1]:
                    break
                stack.pop()
                if not stack:
                    return text[start_idx:idx + 1]

        return None

    def parse_candidate(candidate):
        stripped_candidate = candidate.strip()
        try:
            return json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            if not ((stripped_candidate.startswith("{") and stripped_candidate.endswith("}"))
                    or (stripped_candidate.startswith("[") and stripped_candidate.endswith("]"))):
                raise

            normalized_candidate = (stripped_candidate
                                    .replace("\u2011", "-")
                                    .replace("\u2013", "-")
                                    .replace("\u2014", "-"))
            try:
                parsed_python_literal = ast.literal_eval(normalized_candidate)
                if isinstance(parsed_python_literal, (dict, list)):
                    return parsed_python_literal
            except (SyntaxError, ValueError):
                pass
        sanitized_candidate = (stripped_candidate
                               .replace("\u2011", "-")
                               .replace("\u2013", "-")
                               .replace("\u2014", "-")
                               .replace("\u201c", '"')
                               .replace("\u201d", '"')
                               .replace("\u2018", "'")
                               .replace("\u2019", "'")
                               .replace("\x00", ""))
        sanitized_candidate = escape_control_chars_in_strings(sanitized_candidate)
        sanitized_candidate = remove_trailing_commas(sanitized_candidate)
        return json.loads(sanitized_candidate, strict=False)

    try:
        return parse_candidate(strip_code_fences(content))
    except (json.JSONDecodeError, SyntaxError, ValueError):
        stripped_content = strip_code_fences(content)
        balanced_candidate = extract_balanced_json_substring(stripped_content)
        if balanced_candidate:
            return parse_candidate(balanced_candidate)

        start_idx = stripped_content.find("{")
        end_idx = stripped_content.rfind("}")
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            return parse_candidate(stripped_content[start_idx:end_idx + 1])

        raise


def _supports_json_mode_error(exc):
    error_text = str(exc).lower()
    return any(token in error_text for token in [
        "response_format",
        "json_object",
        "json schema",
        "unsupported",
        "extra_forbidden",
    ])


def _supports_reasoning_effort_error(exc):
    error_text = str(exc).lower()
    return any(token in error_text for token in [
        "reasoning_effort",
        "reasoning effort",
        "extra_forbidden",
        "unsupported",
        "unknown parameter",
        "unexpected keyword",
    ])


def _create_chat_completion(client, request_kwargs, use_json_mode=True, use_reasoning_effort=True):
    completion_kwargs = dict(request_kwargs)
    if use_reasoning_effort:
        completion_kwargs["reasoning_effort"] = "low"
    if use_json_mode:
        completion_kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**completion_kwargs)


def get_json_completion(client, model, messages, temperature=0.1, max_tokens=8192, stop=None):
    _set_last_completion_raw("")
    _set_last_completion_error("")
    _set_last_completion_reasoning("")
    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
    }

    response = None
    last_error = None
    attempt_configs = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]

    for use_json_mode, use_reasoning_effort in attempt_configs:
        try:
            response = _create_chat_completion(
                client,
                request_kwargs,
                use_json_mode=use_json_mode,
                use_reasoning_effort=use_reasoning_effort,
            )
            _set_last_completion_raw(_serialize_response(response))
            _set_last_completion_reasoning(_extract_message_reasoning(response))

            content = _extract_message_content(response)
            try:
                return _extract_json_object(content)
            except Exception as exc:
                last_error = exc
                continue
        except Exception as exc:
            last_error = exc
            is_reasoning_error = _supports_reasoning_effort_error(exc)
            is_json_error = _supports_json_mode_error(exc)

            if use_reasoning_effort and not is_reasoning_error and not is_json_error:
                _set_last_completion_error(str(exc))
                raise
            if use_json_mode and not use_reasoning_effort and not is_json_error:
                _set_last_completion_error(str(exc))
                raise
            if not use_json_mode and use_reasoning_effort and not is_reasoning_error:
                _set_last_completion_error(str(exc))
                raise
            if not use_json_mode and not use_reasoning_effort:
                _set_last_completion_error(str(exc))
                raise

    if response is None:
        _set_last_completion_error(str(last_error) if last_error else "Unknown completion error")
        raise RuntimeError("Failed to create chat completion response.")

    _set_last_completion_error(str(last_error) if last_error else "Failed to parse model response as JSON.")
    raise last_error if last_error else RuntimeError("Failed to parse model response as JSON.")
