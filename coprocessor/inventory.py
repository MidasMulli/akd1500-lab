"""Read sysfs and the SDK target descriptor. Does not open the AKD device."""
from pathlib import Path
from collections import Counter
import datetime as dt
import json
import platform
import akida

def main():
    d=akida.AKD1500();m=d.mesh;s=m.np_sram_size
    nps=[dict(id=[q.ident.col,q.ident.row,q.ident.id],types=[str(t) for t in q.types]) for q in m.nps]
    root=Path('/sys/bus/pci/devices/0007:01:00.0')
    def read(p):return p.read_text().strip() if p.exists() else None
    z=dict(captured_utc=dt.datetime.now(dt.timezone.utc).isoformat(),physical_device_opened=False,
       observed_pci=dict(vendor=read(root/'vendor'),device=read(root/'device'),current_link_speed=read(root/'current_link_speed'),current_link_width=read(root/'current_link_width'),driver=(root/'driver').resolve().name if (root/'driver').exists() else None,resources=read(root/'resource')),
       host=dict(machine=platform.machine(),kernel=platform.release()),
       descriptor=dict(status='SDK_CONFIGURED_TARGET_NOT_A_SILICON_CAPACITY_MEASUREMENT',sdk=akida.__version__,desc=d.desc,ip=str(d.ip_version),nps=nps,type_counts=dict(Counter(t for q in nps for t in q['types'])),np_sram={k:getattr(s,k) for k in dir(s) if not k.startswith('_')},dma_event=str(m.dma_event),dma_configuration=str(m.dma_conf),image_converter=str(m.hrc)),
       boundaries=['Sysfs confirms link/driver, not effective payload bandwidth','Descriptor SRAM budgets do not measure physical occupancy or total silicon SRAM','NPU capability types overlap at one physical position; counts must not be summed','Detailed router-hop timing, SRAM port contention and electrical energy are not observed'])
    Path('studies/20261024-pcie-fabric/evidence/INVENTORY.json').write_text(json.dumps(z,indent=2)+'\n');print(json.dumps({k:v for k,v in z.items() if k!='descriptor'}))
if __name__=='__main__':main()
