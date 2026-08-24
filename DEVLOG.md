# FedMed Devlog

## Phase 0 — Environment Setup

**Date:** 2026-08-24

- Attempted WSL2 setup for Linux-based dev environment (motivated by TenSEAL's
  poor Windows compilation support). Hit a corrupted Windows Installer /
  COM registration issue (`REGDB_E_CLASSNOTREG`) that survived a feature
  re-enable + reboot. Decided to abandon WSL2 for now and build natively
  on Windows instead, accepting the risk that TenSEAL (Week 3 of the spec)
  may need a workaround later (separate Python 3.12 env, Docker, or revisiting
  WSL once the Installer issue is fixed).
- System Python is 3.14.6. Verified PyTorch 2.11 officially supports 3.14
  on Windows, but TenSEAL only ships wheels up to Python 3.13 — noted as a
  known future risk, deferred until Week 3.
- Created project venv (`python -m venv venv`) to isolate dependencies.
- GPU: NVIDIA RTX 5050 Laptop (Blackwell, compute capability sm_120).
  Verified real GPU compute (not just detection) works via a CUDA matmul
  smoke test using `torch==2.11.0+cu128`.