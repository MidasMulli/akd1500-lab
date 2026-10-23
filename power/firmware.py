"""MicroPython entry point for a DEDICATED ESP32 recorder, not an existing device.

Upload is intentionally manual. Call run with explicit board-verified I2C pins.
No firmware, GPIO or serial port is changed by the host benchmark tools.
"""
def run(i2c_id,sda_pin,scl_pin,calibration_id,measurement_domain,hz=20):
    import machine,time,json,sys
    from ina228 import INA228
    if not calibration_id or measurement_domain not in ['akd_card_input','host_plus_card_input']:
        raise ValueError('Calibration identifier and explicit electrical boundary required')
    if not 1<=hz<=20:raise ValueError('Recorder rate 1..20 Hz; JSON must fit a 115200-baud UART')
    sensor=INA228(machine.I2C(i2c_id,sda=machine.Pin(sda_pin),scl=machine.Pin(scl_pin),freq=400000))
    print(json.dumps(dict(kind='recorder',schema=1,source='ina228_hardware',synthetic=False,
              calibration_id=calibration_id,measurement_domain=measurement_domain,
              shunt_ohms=sensor.shunt,current_lsb_a=sensor.lsb,shunt_cal=sensor.cal,
              adc_config=0xb482,requested_hz=hz,device_id=sensor.read(0x3f,2))))
    start=previous=time.ticks_us();elapsed=0;sequence=0
    while True:
        before=time.ticks_us();elapsed+=time.ticks_diff(before,previous);previous=before
        try:
            row=sensor.sample();after=time.ticks_us();row.update(sequence=sequence,sensor_elapsed_us=elapsed,read_span_us=time.ticks_diff(after,before))
            print(json.dumps(row));sequence+=1
        except Exception as exc:
            print(json.dumps(dict(kind='recorder_error',error=str(exc),sequence=sequence)));raise
        time.sleep_us(max(0,int(1e6/hz)-time.ticks_diff(time.ticks_us(),before)))
