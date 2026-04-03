"""
preprocess/1/model.py
=====================
Triton Python Backend – chat-template tokeniser.

Responsibilities
----------------
1. Load ``transformers.AutoTokenizer`` for a Llama-3.1-8B-Instruct checkpoint.
2. On each ``execute()`` call:
   a. Decode the raw UTF-8 bytes Triton hands us (TYPE_STRING tensors arrive
      as numpy ``object`` arrays containing ``bytes`` objects).
   b. Wrap each text in the Llama-3 chat template via
      ``apply_chat_template()``.
   c. Tokenise and return INT32 token-id tensors, zero-padded to the longest
      sequence in the batch so they form a rectangular numpy array.

Shared-Memory note
------------------
Triton's Python backend automatically uses POSIX shared memory to move the
output ``INPUT_IDS`` tensor to the next step (``llama_vllm_shim``) **without
copying through the network stack**.  No explicit SHM API calls are needed in
model.py; the runtime handles it transparently via the
``shm_default_byte_size`` parameter in the backend config.

GIL scaling
-----------
Because each ``instance_group`` entry spawns a **separate Python subprocess**
(not a thread), each instance gets its own interpreter and therefore its own
GIL.  CPU-bound tokenisation scales linearly with ``count``.
"""

import json
import os
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    """Entry-point class required by the Triton Python backend."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, args: dict) -> None:
        """Called once when the model instance is loaded."""
        try:
            self._initialize_impl(args)
        except Exception as e:
            import traceback
            msg = f"[preprocess] initialize() FAILED:\n{traceback.format_exc()}"
            print(msg, flush=True)
            raise  # re-raise so Triton marks the model UNAVAILABLE

    def _initialize_impl(self, args: dict) -> None:
        model_config = json.loads(args["model_config"])

        # Pull user-defined parameters from config.pbtxt.
        params = model_config.get("parameters", {})
        model_name = params.get("model_name", {}).get(
            "string_value", "/models/preprocess/tokenizer"
        )
        fallback_name = params.get("fallback_model_name", {}).get(
            "string_value", "hf-internal-testing/llama-tokenizer"
        )
        self.max_length = int(
            params.get("max_length", {}).get("string_value", "4096")
        )

        # Resolve tokenizer source:
        #   1. If model_name is a local path that exists → use it (fast, no HF).
        #   2. Otherwise try model_name as a HF Hub id.
        #   3. If that fails and fallback_name is set → use fallback.
        tokenizer_source = model_name
        if model_name.startswith("/") and not os.path.isdir(model_name):
            print(
                f"[preprocess] Local tokenizer path not found: {model_name!r}. "
                f"Falling back to HF Hub: {fallback_name!r}. "
                "Run scripts/save_tokenizer.py on the host to create the local copy."
            )
            tokenizer_source = fallback_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            use_fast=True,
        )
        # Llama tokeniser has no pad token by default; reuse EOS.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Output dtype: TYPE_INT32 → np.int32.
        # We hardcode this rather than using pb_utils.get_output_config_by_name()
        # which changed signature across Triton versions.
        self.output_dtype = np.int32

    def execute(self, requests: list) -> list:
        """Process a batch of inference requests.

        Triton may fuse multiple client requests into a single ``execute``
        call when dynamic batching is active upstream (at the ensemble level).

        Parameters
        ----------
        requests:
            List of ``pb_utils.InferenceRequest`` objects.

        Returns
        -------
        list of ``pb_utils.InferenceResponse``
        """
        responses = []

        for request in requests:
            # ----------------------------------------------------------------
            # 1. Extract the raw TEXT input.
            #    Shape: [batch, 1]  – each cell is a bytes object.
            # ----------------------------------------------------------------
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            # numpy array of dtype=object, shape [batch, 1]
            raw_texts = text_tensor.as_numpy()

            # Flatten to a list of Python strings.
            texts: list[str] = []
            for row in raw_texts:
                # Each element may be bytes or str depending on Triton version.
                cell = row[0] if hasattr(row, "__len__") else row
                if isinstance(cell, bytes):
                    cell = cell.decode("utf-8")
                texts.append(cell)

            # ----------------------------------------------------------------
            # 2. Apply the Llama-3 chat template.
            #    We treat each text as the sole "user" turn.
            # ----------------------------------------------------------------
            formatted_texts: list[str] = []
            for text in texts:
                messages = [{"role": "user", "content": text}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                formatted_texts.append(formatted)

            # ----------------------------------------------------------------
            # 3. Tokenise (batch), left-pad to longest sequence.
            # ----------------------------------------------------------------
            encoding = self.tokenizer(
                formatted_texts,
                padding="longest",      # pad to longest in this micro-batch
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",    # numpy output
            )
            # input_ids: int64 numpy array [batch, seq_len]
            input_ids = encoding["input_ids"].astype(self.output_dtype)

            # ----------------------------------------------------------------
            # 4. Build the output tensor and response.
            # ----------------------------------------------------------------
            output_tensor = pb_utils.Tensor("INPUT_IDS", input_ids)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self) -> None:
        """Called once when the model instance is unloaded."""
        del self.tokenizer
