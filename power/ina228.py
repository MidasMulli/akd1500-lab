"""Small, transport-independent INA228 recorder driver; TI SLYS021A Rev A.

Hardware not implied by importing or testing this module. The caller owns I2C
wiring, calibration and the actual current-path boundary.
"""
import math

def signed(value,bits):
    return value-(1<<bits) if value&(1<<(bits-1)) else value

class INA228:
    def __init__(self,i2c,address=0x40,shunt_ohms=.015,current_lsb_a=.00001):
        if not (0x40<=address<=0x4f and 0<shunt_ohms<=1 and current_lsb_a>0):
            raise ValueError('Explicit valid address, shunt and current scale required')
        self.i2c,self.address=i2c,address;self.shunt=shunt_ohms;self.lsb=current_lsb_a
        if self.read(0x3e,2)!=0x5449 or self.read(0x3f,2)>>4!=0x228:
            raise RuntimeError('INA228 identity mismatch; no register writes performed')
        self.cal=int(13107.2e6*self.lsb*self.shunt+.5)
        if not 1<=self.cal<=32767:raise ValueError('SHUNT_CAL outside documented range')
        self.write(0x00,0)  # wide shunt range; no reset of device/accumulators
        self.write(0x02,self.cal)
        self.write(0x01,0xb482)  # bus+shunt continuous; 150 us each, 16 averages
        if self.read(0,2)!=0 or self.read(2,2)!=self.cal or self.read(1,2)!=0xb482:
            raise RuntimeError('Configuration readback mismatch')
    def read(self,reg,n):return int.from_bytes(self.i2c.readfrom_mem(self.address,reg,n),'big')
    def write(self,reg,value):self.i2c.writeto_mem(self.address,reg,value.to_bytes(2,'big'))
    def sample(self):
        flags=self.read(0x0b,2)
        if flags&0x0e00:raise RuntimeError('INA228 energy/charge/arithmetic overflow')
        energy=self.read(0x09,5)
        bus=self.read(0x05,3)>>4;current=signed(self.read(0x07,3)>>4,20)
        power=self.read(0x08,3);shunt=signed(self.read(0x04,3)>>4,20)
        return dict(kind='sample',raw_bus=bus,raw_current=current,raw_power=power,
                    raw_energy=energy,raw_shunt=shunt,diag_flags=flags,
                    bus_v=bus*195.3125e-6,current_a=current*self.lsb,
                    power_w=power*3.2*self.lsb,energy_j=energy*16*3.2*self.lsb,
                    shunt_v=shunt*312.5e-9)
