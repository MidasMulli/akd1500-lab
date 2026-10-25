# Replication console wrapper repair, no scientific retry

The first replication launch shell failed before SSH because a local evidence
directory was absent. After creating it, the first new-seed cell completed and
passed both CPU and native saved-array checks. The outer console then attempted
to index the run() return value, which had not been returned, and stopped.
No model trace failed and no cell is rerun. Return the already-saved audit object
from run() and continue from the first unfinished seed/control. Both console
failures are operational failures, separate from completed scientific sessions.
Streams, seeds, source circuit, program, event caps and endpoints stay unchanged.
