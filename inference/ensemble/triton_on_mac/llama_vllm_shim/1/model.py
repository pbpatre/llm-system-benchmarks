"""
llama_vllm_shim/1/model.py
===========================
Triton Python Backend – GGUF LLM inference via llama-cpp-python.

Modes (set via ``backend_mode`` parameter in config.pbtxt)
-----------------------------------------------------------
``dummy``
    Returns a canned string echoing the token count.  Use for fast
    ensemble/SHM plumbing tests without loading any model weights.

``llama_cpp``
    Loads a GGUF file via llama-cpp-python and runs CPU inference.
    Metal is not available inside the Linux Docker container on M3;
    set n_gpu_layers=0 in config.pbtxt for CPU-only operation.
    On the L40S cluster, replace this model directory with the
    official Triton vLLM backend (platform: "vllm").

Shared Memory (SHM) note
-------------------------
The ``input_ids`` INT32 tensor from ``preprocess`` arrives via POSIX
shared memory — zero copy between the two Python backend subprocesses.
This is transparent to model.py; no SHM API calls are needed here.
Size the SHM region with:
    --backend-config=python,shm-default-byte-size=<bytes>
For batch=32, seq=4096, INT32: 32 × 4096 × 4 = 512 KiB → use 2 MiB.
"""

import json
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Entry-point class required by the Triton Python backend."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, args: dict) -> None:
        """Load the backend selected by ``backend_mode``."""
        try:
            self._initialize_impl(args)
        except Exception as e:
            import traceback
            msg = f"[llama_vllm_shim] initialize() FAILED:\n{traceback.format_exc()}"
            print(msg, flush=True)
            raise

    def _initialize_impl(self, args: dict) -> None:
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})

        self.mode = params.get("backend_mode", {}).get("string_value", "dummy")
        self.max_new_tokens = int(
            params.get("max_new_tokens", {}).get("string_value", "256")
        )
        model_path = params.get("model_path", {}).get("string_value", "")
        n_gpu_layers = int(
            params.get("n_gpu_layers", {}).get("string_value", "-1")
        )
        tokenizer_name = params.get("tokenizer_name", {}).get(
            "string_value", "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.llm = None

        if self.mode == "llama_cpp":
            self._init_llama_cpp(model_path, n_gpu_layers, tokenizer_name)
        # "dummy" mode needs no initialisation

    def _init_llama_cpp(
        self, model_path: str, n_gpu_layers: int, tokenizer_name: str
    ) -> None:
        """Load a GGUF model via llama-cpp-python (Metal backend on M3)."""
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed.  Run:\n"
                "  pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal"
            ) from exc

        if not model_path:
            raise ValueError(
                "backend_mode is 'llama_cpp' but model_path is empty. "
                "Set model_path in llama_vllm_shim/config.pbtxt."
            )

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=4096,
            verbose=False,
        )

        # We also need the tokeniser to decode generated token ids → text.
        # llama-cpp-python can decode internally; store for decode fallback.
        try:
            from transformers import AutoTokenizer  # type: ignore
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            self.tokenizer = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def execute(self, requests: list) -> list:
        """Run generation for a batch of requests.

        Each request carries an ``input_ids`` tensor of shape
        ``[batch_size, seq_len]`` (INT32).  We generate ``max_new_tokens``
        new tokens and return the decoded string.
        """
        responses = []

        for request in requests:
            # ----------------------------------------------------------------
            # 1. Unpack input_ids  – shape [batch, seq_len]
            # ----------------------------------------------------------------
            ids_tensor = pb_utils.get_input_tensor_by_name(request, "input_ids")
            input_ids: np.ndarray = ids_tensor.as_numpy()  # int32

            batch_size = input_ids.shape[0]

            # ----------------------------------------------------------------
            # 2. Generate text per sample in the batch.
            # ----------------------------------------------------------------
            generated_texts = []
            for i in range(batch_size):
                seq = input_ids[i]  # shape [seq_len]
                if self.mode == "dummy":
                    text = self._dummy_generate(seq)
                elif self.mode == "llama_cpp":
                    text = self._llama_cpp_generate(seq)
                else:
                    text = f"[unknown backend_mode: {self.mode}]"
                generated_texts.append(text)

            # ----------------------------------------------------------------
            # 3. Pack output as TYPE_STRING tensor  – shape [batch, 1]
            # ----------------------------------------------------------------
            # Triton expects numpy object arrays of bytes for TYPE_STRING.
            out_array = np.array(
                [[t.encode("utf-8")] for t in generated_texts],
                dtype=object,
            )  # shape [batch, 1]

            output_tensor = pb_utils.Tensor("text_output", out_array)
            responses.append(
                pb_utils.InferenceResponse(output_tensors=[output_tensor])
            )

        return responses

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _dummy_generate(self, input_ids: np.ndarray) -> str:
        """Return a deterministic dummy string for plumbing tests.

        The response encodes enough information to verify that:
        - The correct number of tokens arrived (SHM copy was intact).
        - Padding tokens (0) are correctly counted vs real tokens.
        """
        total_tokens = len(input_ids)
        real_tokens = int(np.count_nonzero(input_ids))
        pad_tokens = total_tokens - real_tokens
        return (
            f"[DUMMY] received {total_tokens} tokens "
            f"({real_tokens} real, {pad_tokens} pad). "
            f"First token id: {input_ids[0]}, last: {input_ids[-1]}. "
            f"Replace backend_mode='llama_cpp' to get real generations."
        )

    def _llama_cpp_generate(self, input_ids: np.ndarray) -> str:
        """Generate via llama-cpp-python (Metal on M3).

        llama-cpp-python's ``Llama.__call__`` accepts a list of token ids
        directly, bypassing any internal tokeniser.
        """
        token_list = input_ids.tolist()
        # Strip trailing pad tokens (EOS id = 128001 for Llama-3).
        # llama-cpp-python treats 0 as a valid token; we strip right zeros.
        while token_list and token_list[-1] == 0:
            token_list.pop()

        output = self.llm(
            token_list,
            max_tokens=self.max_new_tokens,
            echo=False,
        )
        # output is a dict: {"choices": [{"text": "...", ...}], ...}
        return output["choices"][0]["text"]

    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Clean up resources."""
        self.llm = None
