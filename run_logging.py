import csv
import datetime
import json
import os
import re
import threading


_thread_local = threading.local()


def sanitize_text(value):
    if not isinstance(value, str):
        return value
    return value.encode("utf-8", errors="replace").decode("utf-8")


def _ensure_invocation_buffer():
    if not hasattr(_thread_local, "invocation_logs"):
        _thread_local.invocation_logs = []


def start_query_logging():
    _thread_local.invocation_logs = []


def finish_query_logging():
    _ensure_invocation_buffer()
    logs = list(_thread_local.invocation_logs)
    _thread_local.invocation_logs = []
    return logs


def normalize_for_json(value):
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if hasattr(value, "_asdict"):
        return {key: normalize_for_json(item) for key, item in value._asdict().items()}
    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_for_json(item) for item in value]
    return sanitize_text(str(value))


def to_json_text(value):
    return json.dumps(normalize_for_json(value), ensure_ascii=False)


def extract_reasoning(response_json):
    try:
        from model_api import get_last_completion_reasoning
        llm_reasoning = get_last_completion_reasoning()
        if llm_reasoning:
            return llm_reasoning
    except Exception:
        pass

    if not isinstance(response_json, dict):
        return ""

    reasoning_keys = [
        "reasoning",
        "thought",
        "thinking",
        "reflection",
        "debug_thought",
        "analysis",
    ]
    extracted = {key: response_json[key] for key in reasoning_keys if key in response_json}
    if extracted:
        return extracted
    return ""


def record_llm_invocation(prompt, invoker_name, output, reasoning, debug_info=None):
    _ensure_invocation_buffer()
    log_entry = {
        "prompt": prompt,
        "invoker_name": invoker_name,
        "output": output,
        "reasoning": reasoning,
    }
    if debug_info:
        log_entry["debug_info"] = debug_info
    _thread_local.invocation_logs.append(log_entry)


def _sanitize_filename_component(value):
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    normalized = normalized.strip("._")
    return normalized or "unnamed"


def get_run_phase_name(is_searching):
    return "search" if is_searching else "evaluate"


def _get_attempt_runs_dir(script_dir, run_group_name, attempt_index):
    run_group_slug = _sanitize_filename_component(run_group_name)
    return os.path.join(script_dir, "runs", run_group_slug, f"attempt_{attempt_index}")


def create_solution_run_temp_path(script_dir, run_group_name, attempt_index, archive_name, is_searching, timestamp=None):
    phase_name = get_run_phase_name(is_searching)
    run_timestamp = timestamp or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    archive_slug = _sanitize_filename_component(archive_name)
    runs_dir = _get_attempt_runs_dir(script_dir, run_group_name, attempt_index)
    temp_filename = f".{archive_slug}_{run_timestamp}_{phase_name}.tmp.csv"
    return os.path.join(runs_dir, temp_filename), run_timestamp


def finalize_solution_run_csv(temp_path, script_dir, run_group_name, attempt_index, archive_name, is_searching, metric_name, metric_values, timestamp):
    phase_name = get_run_phase_name(is_searching)
    archive_slug = _sanitize_filename_component(archive_name)
    metric_slug = _sanitize_filename_component(metric_name)
    metric_score = sum(metric_values) / len(metric_values) if metric_values else 0.0
    score_slug = f"{metric_slug}-{metric_score:.4f}"
    final_filename = f"{archive_slug}_{timestamp}_{score_slug}_{phase_name}.csv"
    final_path = os.path.join(_get_attempt_runs_dir(script_dir, run_group_name, attempt_index), final_filename)
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    os.replace(temp_path, final_path)
    return final_path


def write_solution_run_outputs(script_dir, run_group_name, archive_name, is_searching, metric_name, attempt_rows, attempt_metrics, timestamp=None):
    run_timestamp = timestamp or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    for attempt_index in sorted(attempt_rows.keys()):
        rows = attempt_rows[attempt_index]
        if not rows:
            continue
        temp_path, _ = create_solution_run_temp_path(
            script_dir,
            run_group_name,
            attempt_index,
            archive_name,
            is_searching,
            timestamp=run_timestamp,
        )
        append_run_rows(temp_path, rows)
        finalize_solution_run_csv(
            temp_path,
            script_dir,
            run_group_name,
            attempt_index,
            archive_name,
            is_searching,
            metric_name,
            attempt_metrics.get(attempt_index, []),
            run_timestamp,
        )


def append_run_rows(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["task", "code", "output", "expected_output", "score", "execution_logs"]
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: sanitize_text(value) for key, value in row.items()})