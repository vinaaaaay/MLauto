"""
Comprehensive logging setup for MLauto.

Creates per-run log files in the output directory:
  - logs.txt            — High-level progress (what you see in the console)
  - debug_logs.txt      — Everything: prompts, responses, state, timing
  - llm_calls.jsonl     — Structured log of every LLM prompt/response pair

Also provides an LLM-call logger that captures prompts and responses for debugging.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


# ── Custom log level: DETAIL (between DEBUG=10 and INFO=20) ──
DETAIL_LEVEL = 15
logging.addLevelName(DETAIL_LEVEL, "DETAIL")


def detail(self, msg, *args, **kw):
    if self.isEnabledFor(DETAIL_LEVEL):
        kw.setdefault("stacklevel", 2)
        self._log(DETAIL_LEVEL, msg, args, **kw)


logging.Logger.detail = detail  # type: ignore


def configure_logging(output_dir: str, verbosity: int = 2) -> None:
    """
    Set up logging handlers for both console and file output.

    Args:
        output_dir: Directory where log files will be written.
        verbosity:
            0 = ERROR only
            1 = WARNING
            2 = INFO (default)
            3 = DETAIL
            4 = DEBUG (everything)
    """
    level_map = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: DETAIL_LEVEL,
        4: logging.DEBUG,
    }
    console_level = level_map.get(verbosity, logging.INFO)

    os.makedirs(output_dir, exist_ok=True)

    handlers = []

    # ── Console handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console_handler.setFormatter(console_fmt)
    handlers.append(console_handler)

    # ── File: logs.txt (mirrors console) ──
    console_file = logging.FileHandler(
        os.path.join(output_dir, "logs.txt"), mode="w", encoding="utf-8"
    )
    console_file.setLevel(console_level)
    file_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_file.setFormatter(file_fmt)
    handlers.append(console_file)

    # ── File: debug_logs.txt (captures EVERYTHING) ──
    debug_file = logging.FileHandler(
        os.path.join(output_dir, "debug_logs.txt"), mode="w", encoding="utf-8"
    )
    debug_file.setLevel(logging.DEBUG)
    debug_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)s:%(funcName)s:%(lineno)d │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    debug_file.setFormatter(debug_fmt)
    handlers.append(debug_file)

    # ── Apply to root logger ──
    logging.basicConfig(
        level=logging.DEBUG,  # root captures everything; handlers filter
        handlers=handlers,
        force=True,
    )


class LLMCallLogger:
    """
    Logs every LLM call (prompt + response) to both:
      - The Python logger (at DEBUG level)
      - A structured JSONL file for post-run analysis

    Usage:
        call_logger = LLMCallLogger(output_dir)
        response = call_logger.call(llm, prompt, node_name="scan_data")
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.jsonl_path = os.path.join(output_dir, "llm_calls.jsonl")
        self.logger = logging.getLogger("mlauto.llm")
        self.call_count = 0

    def call(self, llm, prompt: str, node_name: str = "unknown") -> str:
        """
        Invoke the LLM, log the full prompt and response, and return the response text.

        Args:
            llm: A ChatOpenAI (or compatible) instance.
            prompt: The prompt string to send.
            node_name: Name of the calling node (for log context).

        Returns:
            The response content string.
        """
        self.call_count += 1
        call_id = self.call_count

        self.logger.info(
            f"[Call #{call_id}] {node_name} — sending prompt ({len(prompt)} chars)"
        )
        self.logger.debug(
            f"[Call #{call_id}] {node_name} — PROMPT:\n"
            f"{'='*60}\n{prompt}\n{'='*60}"
        )

        start = time.time()
        response = llm.invoke(prompt)
        elapsed = time.time() - start
        content = response.content

        self.logger.info(
            f"[Call #{call_id}] {node_name} — received response "
            f"({len(content)} chars, {elapsed:.1f}s)"
        )
        self.logger.debug(
            f"[Call #{call_id}] {node_name} — RESPONSE:\n"
            f"{'='*60}\n{content}\n{'='*60}"
        )

        # Write structured JSONL record
        record = {
            "call_id": call_id,
            "timestamp": datetime.now().isoformat(),
            "node": node_name,
            "prompt_length": len(prompt),
            "response_length": len(content),
            "elapsed_seconds": round(elapsed, 2),
            "prompt": prompt,
            "response": content,
        }
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"Failed to write LLM call log: {e}")

        return content


def log_state_snapshot(state: dict, label: str, output_dir: str) -> None:
    """
    Save a snapshot of the current state dict to a JSON file.
    Useful for debugging state transitions between nodes.
    """
    logger = logging.getLogger("mlauto.state")

    # Log summary to console
    keys_with_values = [k for k, v in state.items() if v]
    logger.info(f"State snapshot [{label}]: keys with values = {keys_with_values}")

    # Save full snapshot to file
    snapshots_dir = os.path.join(output_dir, "state_snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%H%M%S")
    safe_label = label.replace(" ", "_").replace("/", "_")
    snapshot_path = os.path.join(snapshots_dir, f"{timestamp}_{safe_label}.json")

    # Make state JSON-serializable
    serializable = {}
    for k, v in state.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            # Truncate very long strings for the snapshot
            if isinstance(v, str) and len(v) > 2000:
                serializable[k] = v[:2000] + f"... [TRUNCATED, total {len(v)} chars]"
            else:
                serializable[k] = v
        else:
            serializable[k] = str(v)

    try:
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        logger.debug(f"State snapshot saved to {snapshot_path}")
    except Exception as e:
        logger.warning(f"Failed to save state snapshot: {e}")
