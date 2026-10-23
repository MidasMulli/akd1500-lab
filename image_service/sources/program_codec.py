"""Offline ProgramInfo codec for installed SDK2.19.3 FNP3 single-pass images.
Schema follows the saved generated headers. Reject unsupported FNP2/skip tracks.
No device API or driver import. Existing C++ verifiers validate emitted images.
"""
import struct
def u(b,p,fmt='I'):return struct.unpack_from('<'+fmt,b,p)[0]
class Table:
 def __init__(self,b,p):self.b=b;self.p=p;self.v=p-u(b,p,'i')
 def field(self,n):
  pos=4+2*n
  if pos>=u(self.b,self.v,'H'):return 0
  k=u(self.b,self.v+pos,'H');return self.p+k if k else 0
 def scalar(self,n,fmt='I'):
  p=self.field(n);return u(self.b,p,fmt) if p else 0
 def ptr(self,n):
  p=self.field(n);return p+u(self.b,p) if p else 0
 def table(self,n):
  p=self.ptr(n);return Table(self.b,p) if p else None
 def vector(self,n,fmt='I'):
  p=self.ptr(n)
  return [u(self.b,p+4+i*struct.calcsize(fmt),fmt) for i in range(u(self.b,p))] if p else None
 def tables(self,n):
  p=self.ptr(n);return [Table(self.b,p+4+i*4+u(self.b,p+4+i*4)) for i in range(u(self.b,p))] if p else None
 def track(self,n):
  p=self.field(n);return list(struct.unpack_from('<2I',self.b,p)) if p else None
 def structs(self,n,size,fmt):
  p=self.ptr(n);return [list(struct.unpack_from('<'+fmt,self.b,p+4+size*i)) for i in range(u(self.b,p))] if p else None
def record(t):
 assert t.field(1)==0,'FNP2 not supported'
 return t.structs(0,8,'2I')
def read(image):
 size=u(image,0)+4;assert size<len(image) and size+u(image,size)+4==len(image)
 r=Table(image,4+u(image,4));assert u(image,r.v,'H')<=50,'new fields require review'
 v=r.ptr(0);version=bytes(image[v+4:v+4+u(image,v)]).decode();d=r.table(1)
 result=dict(version=version,device=[d.scalar(i,'B') for i in range(4)],input_dims=r.vector(2),output_dims=r.vector(3),passes=[],learning=None,epg=r.structs(13,8,'2I'),dummy=r.track(16))
 for n,fmt in [(4,'b'),(5,'b'),(6,'B'),(7,'B'),(8,'I'),(9,'I'),(12,'B'),(17,'B'),(18,'B'),(19,'B'),(22,'B')]:result['s'+str(n)]=r.scalar(n,fmt)
 for n,fmt in [(10,'i'),(11,'f'),(20,'f'),(21,'B')]:result['v'+str(n)]=r.vector(n,fmt)
 for p in r.tables(14):
  assert not p.ptr(1) and not p.ptr(2),'skip tracks need explicit review'
  result['passes'].append([record(x) for x in p.tables(0)])
 l=r.table(15)
 if l:
  np=l.field(3);result['learning']=dict(learning=l.track(0),inference=l.track(1),ram=record(l.table(2)),np=list(image[np:np+3]))
 return result,bytes(image[size:])
def build(info,data):
 import flatbuffers
 b=flatbuffers.Builder(4096)
 def vector(vals,fmt):
  if vals is None:return 0
  f={'I':b.PrependUint32,'i':b.PrependInt32,'f':b.PrependFloat32,'B':b.PrependUint8}[fmt];size=struct.calcsize(fmt);b.StartVector(size,len(vals),size)
  for v in reversed(vals):f(v)
  return b.EndVector()
 def offsets(vals):
  b.StartVector(4,len(vals),4)
  for x in reversed(vals):b.PrependUOffsetTRelative(x)
  return b.EndVector()
 def structs(vals):
  if vals is None:return 0
  b.StartVector(8,len(vals),4)
  for a,c in reversed(vals):b.PrependUint32(c);b.PrependUint32(a)
  return b.EndVector()
 def track_slot(slot,v):
  if v is None:return
  b.Prep(4,8);b.PrependUint32(v[1]);b.PrependUint32(v[0]);b.PrependStructSlot(slot,b.Offset(),0)
 def rec(vals):
  vs=structs(vals);b.StartObject(2)
  if vs:b.PrependUOffsetTRelativeSlot(0,vs,0)
  return b.EndObject()
 passes=[]
 for p in info['passes']:
  rs=offsets([rec(x) for x in p]);b.StartObject(3);b.PrependUOffsetTRelativeSlot(0,rs,0);passes.append(b.EndObject())
 ps=offsets(passes);learn=0
 if info['learning']:
  l=info['learning'];ram=rec(l['ram']);b.StartObject(4);b.PrependUOffsetTRelativeSlot(2,ram,0);track_slot(0,l['learning']);track_slot(1,l['inference']);b.Prep(1,3)
  for x in reversed(l['np']):b.PrependUint8(x)
  b.PrependStructSlot(3,b.Offset(),0);learn=b.EndObject()
 version=b.CreateString(info['version']);b.StartObject(4)
 for n,v in enumerate(info['device']):b.PrependUint8Slot(n,v,0)
 dev=b.EndObject();ind=vector(info['input_dims'],'I');outd=vector(info['output_dims'],'I');epg=structs(info['epg']);vectors={n:vector(info['v'+str(n)],fmt) for n,fmt in [(10,'i'),(11,'f'),(20,'f'),(21,'B')]}
 b.StartObject(23)
 for n,v in [(0,version),(1,dev),(2,ind),(3,outd),(13,epg),(14,ps),(15,learn),*vectors.items()]:
  if v:b.PrependUOffsetTRelativeSlot(n,v,0)
 for n,fmt in [(4,'b'),(5,'b'),(6,'B'),(7,'B'),(8,'I'),(9,'I'),(12,'B'),(17,'B'),(18,'B'),(19,'B'),(22,'B')]:
  {'b':b.PrependInt8Slot,'B':b.PrependUint8Slot,'I':b.PrependUint32Slot}[fmt](n,info['s'+str(n)],0)
 track_slot(16,info['dummy']);root=b.EndObject();b.FinishSizePrefixed(root);return bytes(b.Output())+data
def words(data,span):
 offset,n=span;assert offset%4==0 and offset>=4 and offset+4*n<=len(data)
 return list(struct.unpack_from('<'+str(n)+'I',data,offset))
