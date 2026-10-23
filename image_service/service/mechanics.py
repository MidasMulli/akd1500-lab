from native_client import NativeClient,save
import sys

name,n=sys.argv[1],int(sys.argv[2]);assert n in [1,8,64]
client=NativeClient(name,'mechanics');failed=True
try:
    cycles=[];prior=client.snapshot()
    for k in range(n):
        q=client.query(probe='--probe' in sys.argv and k%2==1);protected=client.snapshot();assert prior['words']==protected['words'],'protected query changed RAM'
        obs=client.observe(2 if k%2==0 else 0);prior=client.snapshot()
        cycles.append(dict(query=q['id'],observe=obs['id'],outcome=2 if k%2==0 else 0))
    save(client.d/'MECHANICS.json',dict(status='PASS',synthetic_bridge_only=True,cycles=cycles,
         fixed_output_slabs=client.ready['slabs'],allocated_bytes=client.ready['allocated_bytes'],peak_bytes=client.ready['peak_bytes']))
    failed=False
finally:client.close(failed=failed)
