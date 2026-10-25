# Correct program-buffer lifetime before requalification

The bounded debugger replay locates SIGSEGV in ProgramInfo::input_dims called by
HardwareDeviceImpl::enqueue. The new benchmark loaded its program byte vector
inside a setup-only scope; that vector was destroyed before enqueue, while the
runtime retained program-data views. The qualified parent keeps the vector alive
throughout execution. The repair restores that same lifetime. Hardware program,
fixture bytes, native algorithm, workload and timing boundaries are unchanged.

Both failed attempts are preserved, with no usable event trace and no admitted
timing sample. The debugger run is diagnostic only. Fresh CPU/native full-state
qualification uses repetition suffix r1; no original directory is overwritten.
Gate lookup accepts only saved passing audits. Clear the blocked state explicitly
with this disposition, then rerun qualification before final-only timing expansion.
