"""
postprocess/1/model.py
======================
Triton Python Backend – output cleaner for Llama-3 chat models.

What it fixes
-------------
Llama-3's chat template generates output that starts with the role header
tokens decoded as text. When llama-cpp-python decodes with ``echo=False``,
the raw output typically looks like:

    "assistant\\n\\nThe capital of France is Paris."
    "assistant\\n\\n2 + 2 = 4"

This backend strips:
  1. The ``assistant\\n\\n`` role header prefix.
  2. Any Llama-3 special token strings (``<|eot_id|>``, ``<|end_of_text|>``).
  3. Leading/trailing whitespace.

The result is clean, client-ready text:
    "The capital of France is Paris."
    "2 + 2 = 4"
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, args: dict) -> None:
        try:
            self._initialize_impl(args)
        except Exception:
            import traceback
            print(f"[postprocess] initialize() FAILED:\n{traceback.format_exc()}", flush=True)
            raise

    def _initialize_impl(self, args: dict) -> None:
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        # Stop strings: truncate output at any of these markers.
        stop_strings_raw = params.get("stop_strings", {}).get(
            "string_value", "<|eot_id|>,<|end_of_text|>"
        )
        self.stop_strings: list[str] = [
            s.strip() for s in stop_strings_raw.split(",") if s.strip()
        ]

        # Whether to strip "assistant\n\n" role header prefix.
        strip_raw = params.get("strip_role_header", {}).get("string_value", "true")
        self.strip_role_header: bool = strip_raw.lower() == "true"

        # Known role header prefixes emitted by Llama-3 chat template.
        # The model echoes the role header as plain text before the response.
        self.role_headers: list[str] = [
            "assistant\n\n",
            "assistant\n",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
        ]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def execute(self, requests: list) -> list:
        responses = []

        for request in requests:
            # ----------------------------------------------------------------
            # 1. Unpack RAW_OUTPUT — shape [batch, 1], dtype object (bytes)
            # ----------------------------------------------------------------
            raw_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_OUTPUT")
            raw_array = raw_tensor.as_numpy()   # object array [batch, 1]

            # ----------------------------------------------------------------
            # 2. Clean each string in the batch.
            # ----------------------------------------------------------------
            cleaned = []
            for row in raw_array:
                cell = row[0] if hasattr(row, "__len__") else row
                if isinstance(cell, bytes):
                    cell = cell.decode("utf-8")
                cleaned.append(self._clean(cell))

            # ----------------------------------------------------------------
            # 3. Pack as TYPE_STRING output — shape [batch, 1]
            # ----------------------------------------------------------------
            out_array = np.array(
                [[t.encode("utf-8")] for t in cleaned], dtype=object
            )
            output_tensor = pb_utils.Tensor("GENERATED_TEXT", out_array)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    # ------------------------------------------------------------------
    # Cleaning logic
    # ------------------------------------------------------------------

    def _clean(self, text: str) -> str:
        """Apply all cleaning steps to a single generated string."""

        # Step 1: strip role header prefix (Llama-3 chat template artifact).
        if self.strip_role_header:
            for header in self.role_headers:
                if text.startswith(header):
                    text = text[len(header):]
                    break   # only strip the first matching header

        # Step 2: truncate at stop strings.
        for stop in self.stop_strings:
            idx = text.find(stop)
            if idx != -1:
                text = text[:idx]

        # Step 3: strip surrounding whitespace.
        text = text.strip()

        return text

    # ------------------------------------------------------------------

    def finalize(self) -> None:
        pass
