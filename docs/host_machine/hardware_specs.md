# Hardware Specifications - Training Machine (adv)

## 1. System Overview
*   **Hostname:** `adv` (Command: `hostname`)
*   **Operating System:** Ubuntu 22.04.5 LTS (Command: `hostnamectl`)
*   **Kernel Version:** Linux 6.8.0-101-generic (Command: `uname -r`)
*   **Architecture:** x86-64

## 2. GPU (Graphics Processing Unit)
*   **Model:** NVIDIA RTX 6000 Ada Generation
*   **VRAM:** 49,140 MiB (48GB)
*   **Driver Version:** 590.48.01
*   **CUDA Version (System):** 13.1
*   **Power Cap:** 300W
*   **Command:** `nvidia-smi`

## 3. CPU (Central Processing Unit)
*   **Model:** 13th Gen Intel(R) Core(TM) i7-13700E
*   **Total Cores:** 16 (8 Performance-cores / 8 Efficient-cores)
*   **Total Threads:** 24
*   **Max Speed:** 5100.00 MHz
*   **Command:** `lscpu`

## 4. Memory (RAM)
*   **Total System RAM:** 31 GiB (~32GB)
*   **Swap:** 2.0 GiB
*   **Command:** `free -h`

## 5. Storage
*   **Primary Drive (OS/Home):** 476.9G (SATA SSD)
*   **Model:** CVB-CD512
*   **Command:** `lsblk -d -o NAME,SIZE,MODEL,ROTA`

---

## Maintenance Commands Reference

### Real-time GPU Monitoring
To monitor GPU temperature, memory usage, and power consumption during training:
```bash
watch -n 1 nvidia-smi