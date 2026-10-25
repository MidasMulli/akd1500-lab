# Power rig: remaining build, October25

**Operator-confirmed inventory:** only a spare ESP32-S3, currently attached to
the Nano. No current monitor, measurement interposer or multimeter on hand.
Do not reflash the connected board until its exact board/pins and serial identity
are selected and the measurement hardware is ready. Software tests do not imply
an assembled or calibrated rig.

| Need | Quantity | Selection / purpose |
|---|---:|---|
| [Adafruit INA228 breakout, product5832](https://www.adafruit.com/product/5832) |1|20-bit current/power/energy recorder; nominal15mOhm shunt. Verify actual board/shunt before calibration.|
| M.2 power-isolating measurement interposer |1|Must support the installed card's connector,2230 mounting if it is the current BrainChip module, and Gen2x2 signal path. All positive card supply paths must pass through the shunt; a passive extension is insufficient.|
| Reference digital multimeter |1|DC voltage and current ranges covering the actual card/load; retain instrument model, resolution and accuracy specifications. Low-current resolution matters more than the displayed digit count.|
| Known calibration load |1|Resistive/electronic load covering idle and working currents; adequate power rating and DMM-verified current.|
| Wiring, terminals and sense leads |set|Short current-rated power conductors, Kelvin sense pair, shared ground,3.3V I2C connection and USB data cable.|
| Whole-host measurement channel |optional second|Needed for a matched end-to-end CPU/AKD energy comparison; do not compare card-only joules with the entire CPU host.|

BrainChip describes its current module as2230 with B+M key in the
[official M.2 announcement](https://investor.brainchip.com/press/brainchip-akd1500-now-available-in-compact-m-2-form-factor-enabling-fanless-edge-ai-in-industrial-and-commercial-designs/).
That is the product-family specification, **not a physical inspection of the
installed card or its power pin routing**. Sysfs currently reports5.0GT/s,x2;
PCIe identification cannot verify mounting or supply isolation. Confirm its label,
connector and carrier before selecting the interposer or fabricating a small
power-breakout PCB. Preserve data/clock/reset/ground routing and power sequencing.

A commercial alternative is the
[Quarch QTL2388 M-key injection fixture](https://quarch.com/products/gen4-m-2-m-key-vertical-injection-fixture/)
with its specified compatible power module. It is **not** a confirmed drop-in
INA228 breakout or a verified purchase choice for this installed carrier. The
DIY route needs a separately verified interposer and supply path. No product
has been ordered, and no card wiring has been changed.

## Finish sequence

1. Identify the spare ESP32-S3 board, permitted I2C pins, and its USB serial
   identifier. Attach the INA228 on the bench, powered separately from the
   measured supply. Confirm manufacturer/device ID and ADC/calibration readbacks.
2. With power removed, verify complete positive-supply interception and absence
   of a bypass/backfeed. Do not hot-plug or modify the installed module live.
3. Before attaching the AKD, record zero plus at least two known-load points
   against the DMM. Measure voltage drop and shunt resistance/temperature;
   document uncertainty and reject insufficient idle-current resolution.
4. Qualify serial continuity, accumulator overflow/reset rejection, and
   counter-versus-integrated-power agreement. Fill calibration.example.json
   using real measurements. Never change its false gates to admit a claim.
5. Start with long bracketing intervals: idle/work/work/idle, repeated in rotated
   order, same exact task and event count. Include setup separately. Existing
   firmware records20Hz; that is appropriate for long intervals, not individual
   microsecond inferences. A GPIO phase trigger remains an **unbuilt/unvalidated**
   addition if short-event synchronization is required.

At the software default10uA current LSB, the TI conversion formulas give32uW
per power count and512uJ per energy count. These are **derived quantization
steps**, not measured accuracy or minimum detectable workload energy. The
150us bus/shunt conversions with16 averages give a nominal4.8ms conversion
cycle;20Hz serial output is slower still. Aggregate repeated operations rather
than assigning counter differences to a single learning event. Formula source:
[TI INA228 RevA datasheet](https://www.ti.com/lit/ds/symlink/ina228.pdf).

Report card gross, powered-idle, incremental and host-plus-card energy separately.
The current blockers are physical hardware, compatible supply interception and
calibration; another software benchmark cannot establish energy efficiency.
