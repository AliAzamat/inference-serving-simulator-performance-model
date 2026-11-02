# Inference and AI Compute Engineer | LLM Serving Simulator and Performance Model

An intermediate Python practice project built from how four postings describe the work: Together AI's Software Engineer, New Grad (2027), which lists "performance optimization" and placement on an Inference team; Nuro's Software Engineer, AI Platform - New Grad, which asks for "software performance tuning and optimization" and different compute modalities (CPU, GPU, FPGA); SpaceX's Software Engineer, Inference (AI Data Engineering), which asks you to "optimize latency and throughput of model inference" with "batch scheduling, global KV cache, and continuous batching"; and OpenAI's Performance Modeling Engineer, which asks for the "ability to reason about performance bottlenecks and scaling behavior" and to "validate models against real system measurements". You build a deterministic simulator of an LLM serving system with a cost model where prefill grows with prompt tokens and each decode step grows with the tokens held in the KV cache. You start from a one-request-at-a-time baseline and measure how it collapses, add continuous batching, add a KV-cache memory budget with an admission policy and compare first-come-first-served against skip-ahead, then load test with a seeded Poisson arrival stream and read p50 latency, p99 latency and tokens per second off a sweep to find the knee. Then comes the part the OpenAI posting names: a closed-form performance model written on paper that predicts batch size, step time, latency and the maximum request rate before the simulator runs, a written predictions file, and a comparison that shows where the model holds (median latency and throughput) and where it misses (the p99 tail). You calibrate the tail with a factor measured on some rates and check the resulting capacity plan on twenty seeds the calibration never saw, because one fresh seed can be a lucky draw. Everything runs on the Python standard library with no GPU and no model. The numbers come from a made-up cost model chosen for this project, not from any company's hardware, and the stack is this project's own choice where the postings do not name one.

Built step-by-step with [KhwajaLabs Build](https://khwajalabs.com).

## Stack
- Python
- discrete-event simulation
- continuous batching
- KV cache
- queueing
- percentiles
- unittest
