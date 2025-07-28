# nano-sglang: Development TODO

This document outlines the development plan and tracks the progress of each core module for the `nano-sglang` backend.

---

## 📦 Module 1: API Server
**[Status: Not Started]**

- [ ] Set up a web server using FastAPI.
- [ ] Implement OpenAI-compatible endpoints: `/v1/chat/completions` and `/v1/completions`.
- [ ] Support both streaming and non-streaming responses.
- [ ] Implement robust request parsing and validation into an internal `Request` object.
- [ ] Create a thread-safe mechanism to push incoming `Request` objects to the Core Scheduler's queue.

---

## 🧠 Module 2: Core Scheduler
**[Status: Not Started]**

- [ ] Implement the main data structures for managing request lifecycle: `waiting_queue`, `running_queue`, `finished_pool`.
- [ ] Build the main inference loop.
- [ ] Implement the continuous batching logic to dynamically create execution batches.
- [ ] Coordinate calls to the RadixEngine for input preparation and the Model Runner for execution.
- [ ] Implement token sampling logic (Greedy, Top-P, Top-K) to process model outputs.
- [ ] Manage the state update for each request after every iteration.

---

## 🌳 Module 3: RadixEngine (KV Cache Manager)
**[Status: Not Started]**

- [ ] **Physical Memory:** Implement a `BlockManager` to manage a pool of fixed-size KV cache blocks on the GPU.
- [ ] **Logical Structure:** Implement the `RadixTree` data structure in Python.
- [ ] **Core Operations:**
    - [ ] `fork()`: Save a sequence's KV cache block mapping to a Radix tree node.
    - [ ] `lookup()`: Find the longest shared prefix for a new sequence and retrieve its block mapping.
    - [ ] `append()`: Allocate a new block for a newly generated token and update the tree.
- [ ] Implement the logic to generate `block_tables` for the model runner based on the current execution batch.

---

## 🚀 Module 4: Model Runner & GPU Kernels
**[Status: Not Started]**

- [ ] Implement a module to load HuggingFace models and weights onto the GPU.
- [ ] **(Challenge)** Write a custom `PagedAttention` CUDA kernel using Triton. This kernel must accept `block_tables` to handle non-contiguous memory access.
- [ ] Wrap the entire model forward pass, integrating the custom PagedAttention kernel.
- [ ] Ensure the model runner correctly handles `token_ids`, `positions`, and `block_tables`.

---

## 🔡 Module 5: Tokenizer Manager
**[Status: Not Started]**

- [ ] Implement a manager to load and cache tokenizers from the HuggingFace Hub based on model names.
- [ ] Provide simple, thread-safe `encode` and `decode` services to the API Server.