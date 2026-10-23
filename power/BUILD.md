# Direct card-power recorder

Status: software and bench design; **no assembled/calibrated power rig yet**.
The PCIe AKD1500 enumerates without a SocDriver power meter. Nano INA3221
readings describe module rails and are not a substitute for M.2 card power.

## Parts and electrical boundary

* One [Adafruit INA228 breakout](https://www.adafruit.com/product/5832), which
  includes a nominal 15 milliohm shunt. A second channel is useful for the upstream
  host-plus-card supply. This is card-input power, not isolated AKD silicon power.
* A dedicated ESP32 board running MicroPython, 3.3 V I2C and a USB data cable.
  Existing ESP/Cardputer experiments are not reflashed by these tools.
* An M.2 PCIe measurement interposer compatible with the actual card key, length,
  lane count and link generation, exposing the complete card supply branch.
  A plain extension cable that leaves parallel power paths intact is insufficient.
  Confirm the installed card and interposer pinout before assigning connector pins.
* Short current-rated conductors/terminals, Kelvin sense connections, a reference
  DMM and a known resistive/electronic calibration load.

Functional wiring:

```
carrier card-supply branch -- IN+ [15 mOhm shunt] IN- -- all card supply pins
                                             |        AKD M.2 module
                                        load-side bus voltage
carrier/card ground --------------------------+----------------------

dedicated ESP32 3.3V/GND/SDA/SCL --> INA228 logic/pullups
dedicated ESP32 USB             --> recorder host
```

Interpose the **entire** positive supply branch; preserve PCIe data, reference
clock, reset and ground routing. Verify with power removed that no parallel
positive supply or peripheral backfeed bypasses the shunt. The monitor logic is
powered separately from the measured card branch. Do not hot-plug or modify the
installed module. This is a wiring plan, not a manufactured, signal-integrity
qualified interposer or an assertion of a verified connector pin map.

## Calibration and admission

1. Verify part identity, shunt value and full supply isolation. Test the recorder
   on a separate known load before connecting the AKD branch.
2. Measure zero offset, then at least two representative currents against a DMM;
   retain reference readings, duration, temperature and shunt tolerance. Verify
   voltage drop stays inside the card's specified supply tolerance.
3. Read back ADC/current-scale configuration. Check sequence continuity,
   conversion timing, monotonic energy and overflow flags.
4. Set measured calibration fields explicitly. The example deliberately refuses
   a hardware energy claim. Compare the hardware accumulator with integrated
   power on long intervals; sensor resolution is not measurement accuracy.
5. Use long ABBA workload/idle phases and a physical synchronization trigger for
   short events. USB reception time alone cannot align individual inferences.

Report card gross energy, powered-idle energy, incremental energy and whole-system
energy separately. Include programming/wakeup costs when relevant. Never compare
card-only energy against CPU-plus-entire-host energy, or use a different model,
accuracy, memory capacity or update budget to claim a 10x benefit.

`ina228.py` implements the [TI INA228 Rev A register conversions](https://www.ti.com/lit/ds/symlink/ina228.pdf).
`firmware.py` requires explicit board-verified I2C pins. `collect.py` captures raw
serial data. `analyze.py` rejects gaps, resets, overflow and uncalibrated claims.
Hardware execution remains unverified until the physical rig is installed.
