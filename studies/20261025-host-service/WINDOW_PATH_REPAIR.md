# Pre-device path repair

The proposed completion-window runner/build names collided with the existing
FNP2 class-window tools. Existing tools are restored byte-for-byte from the
previously untouched Git HEAD. New files use host_window_run.py and
build_host_window.sh. The first10 CPU-only qualification attempts and theirV1
freeze are preserved; there were no physical completion-window attempts before
this repair. All five CPU paths, both arms, must requalify as repetition1 under
WINDOW_FROZEN_V2 before the new native smoke. The actor binary/arithmetic stays
unchanged. Add parent observations of /proc/actor/stat at the common window
boundaries to capture process-wide thread spin between arrivals; per-call CPU
alone can miss that cost. No outcome/selection or timing criterion changes.
