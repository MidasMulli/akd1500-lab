"""Compiler-only reproduction of the documented image -> fabric -> learner path."""
from pathlib import Path
import json
import traceback
import numpy as np
import akida
from tools.prepare import PARAMS, parts
from metal.prepare import mapping

P = Path(__file__).resolve().parents[1]
S = P / 'studies/20261024-pcie-fabric'
SEED = 106101
WIDTHS = [16, 64, 128]


def conv(x, w, threshold):
    """NHWC integer cross correlation, TF SAME padding, stride two, binary ReLU."""
    n, h, width, c = x.shape
    out_h, out_w = (h + 1) // 2, (width + 1) // 2
    ph, pw = max(0, (out_h-1)*2+3-h), max(0, (out_w-1)*2+3-width)
    z = np.pad(x.astype(np.int32), ((0,0),(ph//2,ph-ph//2),(pw//2,pw-pw//2),(0,0)))
    y = np.zeros((n,out_h,out_w,w.shape[-1]), np.int32)
    for ky in range(3):
        for kx in range(3):
            y += z[:,ky:ky+out_h*2:2,kx:kx+out_w*2:2] @ w[ky,kx].astype(np.int32)
    return y if threshold is None else (y > threshold).astype(np.uint8)


class Select(akida.MapConstraints):
    def select_nps(self, source_nps, num_nps, type):
        if type == akida.NP.Type.FNP3:
            found = [q for q in source_nps if [q.col,q.row,q.id] == [3,3,0]]
        else:
            found = [q for q in source_nps if [q.col,q.row,q.id] != [3,3,0]]
        assert len(found) >= num_nps, (str(type),num_nps)
        return found[:num_nps]


def main():
    out = S/'fixtures03'; out.mkdir(exist_ok=False)
    rng = np.random.default_rng(SEED+1)
    x = rng.integers(0,256,(96,32,32,3),np.uint8)
    x[::8] = 0; x[1::8] = 255
    labels = np.arange(96,dtype='<i4') % 2
    results = []
    for width in WIDTHS:
        name = f'rgb32_c{width}'; f=out/name
        try:
            r = np.random.default_rng(SEED)
            w0=r.integers(-3,4,(3,3,3,width),np.int8)
            z0=conv(x[:16],w0,None); t0=np.median(z0,axis=(0,1,2)).astype('<i4')
            a0=conv(x,w0,t0)
            w1=r.integers(-7,8,(3,3,width,8),np.int8)
            # Akida V1 Convolutional is convolution; InputConvolutional is cross correlation.
            z1=conv(a0[:16],w1[::-1,::-1],None);t1=np.median(z1,axis=(0,1,2)).astype('<i4')
            a1=conv(a0,w1[::-1,::-1],t1);assert a1.shape==(96,8,8,8)
            m=akida.Model()
            m.add(akida.InputConvolutional(name='rgb_to_native_events',input_shape=(32,32,3),filters=width,kernel_size=(3,3),kernel_stride=(2,2),weights_bits=8,act_bits=1))
            l=m.get_layer('rgb_to_native_events');l.set_variable('weights',w0);l.set_variable('threshold',t0);l.set_variable('act_step',np.ones(width,np.float32))
            got=m.forward(x);assert np.array_equal(got,a0),('independent HPC integer reference',int(np.sum(got!=a0)))
            m.add(akida.Convolutional(name='native_spatial_features',filters=8,kernel_size=(3,3),kernel_stride=(2,2),weights_bits=4,act_bits=1))
            l=m.get_layer('native_spatial_features');l.set_variable('weights',w1);l.set_variable('threshold',t1);l.set_variable('act_step',np.ones(8,np.float32))
            got=m.forward(x);assert np.array_equal(got,a1),('independent CNP integer reference',int(np.sum(got!=a1)))
            m.save(str(f.with_suffix('.front.fbz')))
            m.add(akida.FullyConnected(name='native_binary_learner',units=128,weights_bits=1,activation=False))
            m.compile(optimizer=akida.AkidaUnsupervised(**PARAMS))
            lw=np.zeros((512,128),np.int8)
            for j in range(128):lw[r.choice(512,32,False),j]=1
            l=m.get_layer('native_binary_learner');l.set_variable('weights',lw.reshape(1,1,512,128));l.set_variable('threshold_learning',np.zeros(128,np.int32))
            # V1 spatial -> FC axon order is channel, column, row (not NHWC).
            features=a1.transpose(0,3,2,1).reshape(96,512).copy()
            y=features.astype(np.int32)@lw.astype(np.int32)
            assert np.array_equal(m.forward(x).reshape(96,128),y),'end-to-end virtual contract'
            m.save(str(f.with_suffix('.fbz')))
            d=akida.AKD1500();m.map(d,hw_only=True,mode=akida.MapMode.Minimal,constraints=Select(d))
            assert len(m.sequences)==len(m.sequences[0].passes)==1,'resident single pass required'
            b=bytes(m.sequences[0].program);ram,npid=parts(b)
            assert len(ram)==2+128*20 and npid==[3,3,0],(len(ram),npid)
            for ext,data in [('bin',b),('ram',np.asarray(ram,'<u4').tobytes()),('inputs',x.tobytes()),('features',features.tobytes()),('expected',y.astype('<i4').tobytes()),('labels',labels.tobytes()),('w0',w0.tobytes()),('w1',w1.tobytes()),('t0',t0.tobytes()),('t1',t1.tobytes())]:
                f.with_suffix('.'+ext).write_bytes(data)
            f.with_suffix('.meta').write_text(f'32 {width} 8 512 128 32 2 96\n')
            row=dict(name=name,status='COMPILER_AND_INDEPENDENT_INTEGER_PASS',mapping=mapping(m),learning_np=npid,input_shape=[32,32,3],features=512,learning_records=128,learning_ram_bytes=len(ram)*4,nominal_dense_macs=16*16*27*width+8*8*9*width*8,feature_density=float(a1.mean()))
        except Exception as e:
            row=dict(name=name,status='FAILED_PRESERVED',error=str(e),traceback=traceback.format_exc())
        results.append(row)
        (out/'BUILD.json').write_text(json.dumps(dict(sdk=akida.__version__,physical_device_opened=False,cells=results),indent=2)+'\n')
        print(json.dumps(row),flush=True)
        if row['status']=='FAILED_PRESERVED':break


if __name__=='__main__':main()
