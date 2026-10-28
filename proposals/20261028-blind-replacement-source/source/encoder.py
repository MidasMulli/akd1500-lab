from pathlib import Path
import numpy as np
from core import Layout

B=Path(__file__).resolve().parents[1]

class Encoder:
    def __init__(self,profile):
        self.g=Layout(profile);self.z=np.load(B/'encoders/PRIOR_PCA14.npz');self.edges=np.array([-1.,0.,1.]if profile=='quad13'else[-2.,-1.,-.5,0.,.5,1.,2.]);self.scale=self.z['singular'][:self.g.d]/np.sqrt(285)
        assert np.all(self.scale>1e-12)
    def project(self,h0,h5):
        a,b=np.asarray(h0,'f8'),np.asarray(h5,'f8');return(np.r_[a,b,b-a]-self.z['mean'])@self.z['basis'][:self.g.d].T
    def encode(self,h0,h5,program):
        v=self.project(h0,h5)/self.scale;levels=np.searchsorted(self.edges,v,side='right');return self.g.key(levels,program),v
