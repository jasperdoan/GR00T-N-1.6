# Environment Management SOP

To keep the system stable, I’ve been following a "Clean Root" policy. I wanted to document this here so we’re on the same page for future installs.

## 1. "Clean Root" Policy
*   **System Root:** Only for hardware drivers, core compilers, and essential libraries (NVIDIA drivers, CUDA Toolkit, FFmpeg, etc.).
*   **User Space:** All project-specific libraries (PyTorch, Transformers, Isaac Lab) should stay inside virtual environments (`venv`, `conda`) or Docker containers.
*   **Why:** This prevents a version update for one project (like LeRobot) from breaking another project (like Isaac Sim).

## 2. Current System-Level Installs (Global)
These are already installed at the system level and are available to all users:

| Tool | Version | Command to verify |
| :--- | :--- | :--- |
| **Python** | 3.10.19 | `python3 --version` |
| **NVIDIA Driver** | 590.48.01 | `nvidia-smi` |
| **CUDA (nvcc)** | 11.5 | `nvcc --version` |
| **Docker** | 29.2.1 | `docker --version` |
| **Git** | 2.34.1 | `git --version` |
| **FFmpeg** | 4.4.2 | `ffmpeg -version` |


> **Note on CUDA:** There is a discrepancy between the Driver's CUDA capability (13.1) and the installed Toolkit (11.5). I've kept it at 11.5 for compatibility with current scripts.


## 3. Current Project Environments
I have already set up the following under my user (`jasper@adv`). If you need to replicate these under `{user}@adv`, I can share my `requirements.txt` or `environment.yaml` files.


## 4. Recommendations for New Setups
If you’re setting up **phospho**, **lerobot**, or any new Python environment, I’d suggest:
1. Creating a new venv: `python3 -m venv .venv`
2. If you need system-level packages (like `libopencv-dev`), please let me know so we can track the install in this doc!

---
> Self-Correction/Feedback: If you think we should move toward a full Docker-based workflow for everything, let me know. I'm happy to help migrate the current venvs.*