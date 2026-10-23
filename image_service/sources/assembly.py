import json,struct,copy
from compiler_helpers import assemble
from program_codec import read,build,words
def two_memory_assembly(prefix1,prefix2,full):
 image,audit2=assemble(prefix2,full,expected_fixed=5);h,data=read(image);one,d1=read(prefix1);lr=one['learning'];ram=words(d1,lr['ram'][0]);inf=words(d1,lr['inference']);lea=words(d1,lr['learning']);primary=lr['np']
 rec=next(f for f in audit2['fixed'] if f['np']==primary);ri=rec['record'];assert len(h['passes'][0][ri])==1
 span=h['passes'][0][ri][0];packet=words(data,span);assert len(packet)==1334 and packet[52:54]==ram[:2]
 ni=packet[:52];nl=ni[:]
 for i in [38,39,40]:assert ni[i]==inf[i];nl[i]=lea[i]
 assert [i for i,(a,b) in enumerate(zip(inf,ni)) if a!=b]==[46,47,49]
 assert [i for i,(a,b) in enumerate(zip(ram,packet[52:])) if a!=b]==[2+20*j+k for j in range(64) for k in [1,2,3]]
 d=bytearray(data)
 newsp=[len(d),len(ni)+len(ram)];d.extend(struct.pack('<'+'I'*(len(ni)+len(ram)),*(ni+ram)));h['passes'][0][ri]=[newsp];struct.pack_into('<I',d,0,len(d)-4)
 result=build(h,bytes(d));hh,dd=read(result);assert words(dd,hh['passes'][0][ri][0])==ni+ram
 lr2=hh['learning'];memory2=dict(np=lr2['np'],initial_words=words(dd,lr2['ram'][0]),inference=words(dd,lr2['inference']),learning=words(dd,lr2['learning']))
 memories=[dict(np=primary,initial_words=ram,inference=ni,learning=nl),memory2]
 fixed=[v for v in audit2['fixed'] if v['np']!=primary];assert len(fixed)==4
 return result,dict(memories=memories,fixed=fixed,program_info_learning_metadata='second memory; first memory mode explicitly programmed from frozen packets',primary_restored_auxiliary=True)
