# nano-sglang

![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

---

### 🎯 Project Goal

**`nano-sglang`** is an educational open-source project dedicated to implementing the **backend inference system** of the high-performance LLM framework, SGLang.

The primary goal is **NOT** to create a full-stack replica of SGLang. Instead, we are focusing exclusively on the core backend components that make SGLang exceptionally fast and efficient. This project serves as a hands-on learning guide for students, researchers, and engineers who want to deeply understand the inner workings of modern LLM inference engines.

Our implementation will be centered around two key architectural pillars:
1.  **Continuous Batching:** To maximize GPU utilization by dynamically batching incoming requests.
2.  **RadixAttention (with PagedAttention):** To achieve state-of-the-art KV cache management and reuse through a combination of a logical Radix Tree and a physical paged memory manager.

The final system will be a functional, high-performance inference server capable of running on NVIDIA GPUs, with an OpenAI-compatible API, released under the Apache 2.0 License.