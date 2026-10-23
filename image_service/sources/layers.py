import itertools
import numpy as np

def prediction_weights(mapping):
 w=np.zeros((512,64),np.int8)
 for c in range(8):
  w[28*c:28*c+28,c]=1
  if mapping[c] is None:w[480:484,c]=1
  else:w[224+4*mapping[c]:228+4*mapping[c],c]=1
 for j in range(8,32):w[480:512,j]=1
 feats=[28*c for c in range(8)]+[224+4*s for s in range(4)]+list(range(256,260))+list(range(260,265))
 for j,f in enumerate(feats):w[[f,*range(480,511)],32+j]=1
 for j in range(len(feats),32):w[480:512,32+j]=1
 return w

def message_encoder():
 w=np.zeros((64,512),np.int8);t=np.zeros(512,np.int32)
 for s in range(4):w[32+2*s,24*s:24*s+24]=w[33+2*s,24*s:24*s+24]=1
 for c in range(8):w[32+c,100:104] = int(c%2==1);w[32+c,96:100]=int(c%2==0)
 # The first query activates all8contexts: suppress the sum of four action0
 # relays, rather than assuming only one current-state context pair.
 w[48,96:100]=-7;w[48,100:104]=1
 for out in range(4):w[40+out,112+4*out:116+4*out]=w[48,112+4*out:116+4*out]=1
 for s in range(4):w[44+s,258+s]=1
 for src,dst in [(48,262),(49,263),(50,264),(51,284),(52,285)]:w[src,dst]=1
 for c in range(8):w[c,270+c]=w[44+c//2,270+c]=1;t[270+c]=1
 for s in range(4):w[2*s:2*s+2,280+s]=1;w[48,280+s]=2;t[280+s]=2
 return w,t

def history_weights(history):
 w=np.zeros((512,64),np.int8)
 for s in range(4):
  w[24*s:24*s+24,s]=1
  if history[s] is None:w[480:488,s]=1
  else:
   a,out=history[s];w[96+4*a:100+4*a,s]=1;w[112+4*out:116+4*out,s]=1
 for j in range(4,32):w[480:512,j]=1
 feats=[24*s for s in range(4)]+[112+4*s for s in range(4)]+[96,100]+list(range(258,262))+[262,263,264]+list(range(270,278))+list(range(280,284))+[284,285]
 assert len(feats)==31
 for j,f in enumerate(feats):w[[f,*range(480,511)],32+j]=1
 w[480:512,63]=1
 return w

def policy(goal,start,twohop):
 w=np.zeros((64,128),np.int8);t=np.zeros(128,np.int32);a=np.zeros((128,32),np.int8)
 for s in range(4):
  w[42+s,s]=w[47,s]=1;t[s]=1;a[s,4+s]=1
  for act in range(2):
   j=4+2*s+act;w[49+2*s+act,j]=w[47,j]=1;t[j]=1;a[j,28+act]=a[j,31]=1
  j=12+s;w[57+s,j]=w[47,j]=1;t[j]=1;a[j,24+s]=1
  for k in range(6):
   j=24+6*s+k;w[42+s,j]=1;w[47,j]=-4
   if k==0:w[61,j]=1;t[j]=1;act=0
   elif k==1:w[62,j]=1;w[61,j]=-4;t[j]=1;act=1
   else:
    w[61:63,j]=-4
    if k==2:w[49+2*s,j]=1;t[j]=1 if twohop else 10;act=0
    elif k==3:w[50+2*s,j]=1;w[49+2*s,j]=-4;t[j]=1 if twohop else 10;act=1
    else:
     if twohop:w[49+2*s:51+2*s,j]=-4
     if k==4:w[s,j]=1;t[j]=1;act=0
     else:w[s,j]=-1;act=1
   a[j,8+2*s+act]=1;a[j,20 if k<2 else 21 if k<4 else 22]=1
 for s,out,act in itertools.product(range(4),range(4),range(2)):
  j=48+8*s+2*out+act;w[32+s,j]=w[36+out,j]=w[40+act,j]=1;w[46,j]=-4;t[j]=2;a[j,start if out==goal else out]=1
 w[48,80]=1;a[80,16]=1
 w[47,81]=1;a[81,17]=1
 w[46,82]=1;w[47,82]=-1;a[82,18]=1
 w[46,83]=-1;t[83]=-1;a[83,19]=1
 w[36+goal,84]=1;w[46,84]=-4;a[84,23]=1
 for s in range(4):w[s,85+s]=w[42+s,85+s]=1;t[85+s]=1;a[85+s,30]=1
 w[41,90]=1;w[46,90]=-4;a[90,16]=1
 return [('native_predecessor_and_two_step_action_logic',w,t),('commit_recurrent_state_and_prediction_caches',a,np.zeros(32,np.int32))]
