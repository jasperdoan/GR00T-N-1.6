# Storage Policy & Data Management

## 1. Checkpoint Cleanup (Best Practice)
Training robotics models might generates very large checkpoint files. To keep the machine healthy, I’ve been following this cleanup routine:

*   **During Training:** I usually keep the 5 most recent checkpoints (e.g., saving every ~2k steps).
*   **Post-Training:** Once a run is finished or validated, I delete the intermediate steps and only keep the final/best checkpoint.
*   **Example:** 
    *   A full set of GR00T N1.6 checkpoints (12k–20k steps) took up **~125 GB**. 
    *   After cleaning up and keeping only the 20k step completion, the size dropped to **~22 GB**.
    *   Please try to prune your `outputs/` or `checkpoints/` folders once a training session is verified.

## 2. Directory Usage
*   **Temporary Files:** Be mindful of `/tmp` and hidden cache folders (like `~/.cache/huggingface`), as these can grow very large very quickly.

## 3. Future Expansion
Waylon mentioned adding an additional SSD to this machine in the future.

## 4. Commands for Storage Monitoring

Check total space is left on the drive:
```bash
df -h /
```