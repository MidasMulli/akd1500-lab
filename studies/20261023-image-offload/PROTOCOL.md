# Real image-service completion-wait comparison

Fresh bounded study; previous image-service study stays closed. Reuse its
validated two-learning-memory controller and real CPU/GStreamer image workers.
The new three-bank association cascade is a separate mechanism study.

Compare the original 20 ms no-I/O guard interval with bounded 50 us sleeps and
read-only inference-completion checks. After release, zero host writes until
completion, identical native actions/outcomes and all learning RAM must match.
No hidden worker retry; only owned workers can be paused. Full raw MMIO audit,
artifact checks and physical-outcome/controller-state separation remain.

Start with two owned JPEG jobs, then one 24-job outage/recovery series and its
identical CPU-algorithm control. Compare exact action/outcome sequences. Keep
service wall time descriptive; worker timeouts dominate, so do not infer an
end-to-end throughput or energy advantage. Count completion-wait time separately.
At most eight sessions, 180 seconds each; check source manifests and fixed
programs. Stop on any unexplained audit or artifact mismatch.
