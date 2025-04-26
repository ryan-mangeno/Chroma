# Chroma

![chroma](https://github.com/user-attachments/assets/cee40a0e-7a78-4044-9ff2-e6d4638ab0d4)


*(Inspired by Chromatic Number)*

---

**Chroma** is a lightweight, high-performance math library designed for use in **2D and 3D applications**, such as **game development**, **graphics programming**, **simulation**, and **other math-intensive projects**.

It provides **SIMD-optimized** vector and matrix types, similar in spirit to libraries like **GLM**, but with a focus on **modern hardware instructions** (like AVX and FMA)

Chroma is intended to be **compiled as a module** (no `main()` function is included) and **linked into your larger application**.

---

## Features

- Fast, SIMD-accelerated math operations
- Support for 2D and 3D vectors
- Matrix types and transformations
- Optimized for modern CPUs 
- Lightweight and self-contained (no external dependencies)

---

## Compilation Instructions (Windows)

**Requirements:**
- [MSYS2](https://www.msys2.org/) (with MinGW-w64 toolchain installed)
- g++ (MinGW-w64, version 14.1.0+ recommended)

**Compile Command:**

```bash
g++ -mavx -mfma -O2 -march=native -c crm_mth.cpp
