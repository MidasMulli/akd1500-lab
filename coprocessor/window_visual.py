"""Data-bound memory/response replay, not electrical tracing or live video."""
from pathlib import Path
import hashlib,json,math,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from coprocessor.window_archive import BASE,P,audit_public,decode
W,H=3840,2160;BG='#080e18';FG='#e5edf5';MUTED='#92a6bc';BLUE='#77b6f4';GREEN='#78c2aa';RED='#ed9589'
FONT=Path('/System/Library/Fonts/Supplemental/Arial.ttf');BOLD=Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
if not FONT.exists():FONT=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf');BOLD=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
fonts={}
def font(s,b=False):
 if (s,b) not in fonts:fonts[s,b]=ImageFont.truetype(str(BOLD if b else FONT),s)
 return fonts[s,b]
def text(d,xy,s,size=32,color=FG,b=False):d.text(xy,s,font=font(size,b),fill=color)
def rgb(c):return tuple(bytes.fromhex(c.lstrip('#')))
def main(video=False):
 summary=audit_public();m=json.loads(BASE.with_suffix('.json').read_text());z=dict(np.load(BASE.with_suffix('.npz'),allow_pickle=False));f=m['fixtures']['s107101'];initial=z['s107101__initial'];u,stride,tile=f['records'],f['record_words'],f['tile_records'];target=f['rows'][0]['target'];label=f['rows'][0]['label'];local=target-label*tile
 direct=next(r for r in m['runs'] if r['mode']=='direct');direct_state=decode(z,direct['name'],initial,1)[0]
 window=next(r for r in m['runs'] if r['name']=='s107101_window_n1');window_state=decode(z,window['name'],initial,2)[0]
 wrong=int(np.flatnonzero(np.any(direct_state.reshape(u,stride)!=initial.reshape(u,stride),axis=1))[0]);before_scores=z[direct['name']+'__pre'];after_scores=z['s107101_window_n1__recall'];cue=z['s107101__inputs'][0];removed=f['rows'][0]['removed_bit'];added=f['rows'][0]['added_bit']
 def bank(scores):
  v=scores.reshape(88,128).astype(float)/32;c=np.zeros((88,128,3),np.uint8)
  for j,(a,b) in enumerate([(22,84),(37,154),(54,182)]):c[:,:,j]=a+(b-a)*v
  c=np.repeat(np.repeat(c,9,axis=0),10,axis=1);c[8::9,:,:]=rgb('#0c1523');c[:,9::10,:]=rgb('#0c1523');return Image.fromarray(c)
 banks=[bank(before_scores),bank(after_scores)]
 bases=[]
 for stage in range(4):
  im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
  text(d,(130,83),'Native learning, past a failing address',84,b=True)
  text(d,(134,205),'AKD1500  /  11,264 allocated records  /  same chip, cue and initial weights',38,MUTED)
  phase=['Same cue. Same starting model.','Direct full-bank learning changes the wrong record.','A bounded class view keeps native learning at the intended record.','Full-model recall and complete memory verification.'][stage]
  text(d,(135,289),phase,36,BLUE)
  for side,x in enumerate([130,1980]):
   y=365;pw=1730;d.rounded_rectangle((x,y,x+pw,1815),radius=28,fill='#101b2a',outline='#2a3e54',width=2)
   color=RED if side==0 else GREEN
   text(d,(x+55,y+38),'DIRECT FULL-BANK PATH' if side==0 else 'BOUNDED CLASS-WINDOW PATH',28,color,b=True)
   text(d,(x+55,y+92),'Same cue, wrong destination' if side==0 else 'Same model, correct native update',54,b=True)
   text(d,(x+55,y+175),'Nano host DDR · logical records · brightness = match response',31,MUTED)
   gx,gy=x+255,y+278
   im.paste(banks[1 if side==1 and stage>=3 else 0],(gx,gy))
   d=ImageDraw.Draw(im);d.rectangle((gx-3,gy-3,gx+1280+2,gy+792+2),outline='#3c536b',width=2)
   text(d,(x+42,gy+128),'CLASS 0',26,MUTED,b=True);text(d,(x+42,gy+166),'5,632 records',25,MUTED)
   text(d,(x+42,gy+524),'CLASS 1',26,MUTED,b=True);text(d,(x+42,gy+562),'5,632 records',25,MUTED)
   d.line((gx,gy+396,gx+1280,gy+396),fill='#7994ae',width=3)
   if side==1 and stage>=1:
    d.rectangle((gx-8,gy+396-6,gx+1287,gy+799),outline=BLUE,width=5)
    text(d,(gx+780,gy+410),'Teacher label 1 selects this view',26,BLUE,b=True)
   def center(index):return gx+(index%128)*10+5,gy+(index//128)*9+4
   tx,ty=center(target);d.ellipse((tx-21,ty-21,tx+21,ty+21),outline=FG,width=4)
   text(d,(tx+34,ty-42),f'Intended record {target}',30,FG,b=True)
   if side==0 and stage>=1:
    ax,ay=center(wrong);d.ellipse((ax-27,ay-27,ax+27,ay+27),outline=RED,width=6)
    text(d,(ax-440,ay-63),f'Actual changes: record {wrong}',31,RED,b=True)
   if side==1 and stage>=2:d.ellipse((tx-30,ty-30,tx+30,ty+30),outline=GREEN,width=7)
   chip=(x+60,y+1130,x+455,y+1290);d.rounded_rectangle(chip,radius=15,fill='#1b2c40',outline=color,width=3)
   for p in range(12):
    px=chip[0]+22+p*30;d.line((px,chip[1]-10,px,chip[1]),fill='#63809c',width=4);d.line((px,chip[3],px,chip[3]+10),fill='#63809c',width=4)
   text(d,(x+95,y+1154),'AKD1500',38,b=True);text(d,(x+95,y+1215),'Native FNP2 learner',29,MUTED)
   text(d,(x+515,y+1108),'Target record connections: 32 selected input bits',29,MUTED)
   state=window_state if side==1 and stage>=2 else initial
   mask=np.unpackbits(state.reshape(u,stride)[target,4:].copy().view('u1'),bitorder='little')
   bx,by=x+515,y+1162
   for k in range(512):
    xx,yy=bx+(k%64)*15,by+(k//64)*15;fill=GREEN if mask[k] else '#23384e'
    d.rectangle((xx,yy,xx+11,yy+11),fill=fill)
   for k,c in [(removed,RED),(added,GREEN)]:
    xx,yy=bx+(k%64)*15,by+(k//64)*15;d.rectangle((xx-3,yy-3,xx+14,yy+14),outline=c,width=2)
   text(d,(bx,by+140),f'Recorded connection swap: input {removed} → input {added}' if side==1 and stage>=2 else 'Captured target connection mask before learning',27,color if side==1 and stage>=2 else MUTED)
   text(d,(x+58,y+1342),'RAM check failed · direct branch halted' if side==0 and stage>=1 else 'Full-model recall: 32 / 32 matches' if side==1 and stage>=3 else f'Teacher class {label} · native match response: 31 / 32',34,color if stage else FG,b=True)
   text(d,(x+58,y+1397),f'Global record {target} → local view record {local}' if side==1 else 'Inference succeeds; the stored learning update fails',27,MUTED)
   if side==1 and stage>=3:text(d,(x+1030,y+1397),'Frozen recall: 31 / 32',27,MUTED)
  text(d,(135,1855),'24 selected, single-swap updates  ·  3 initializations  ·  complete bank checks  ·  frozen + CPU controls',36,b=True)
  text(d,(135,1921),'Host selects a class view from the teacher label. The chip chooses the winner and changes connections.',32,MUTED)
  text(d,(135,1983),'No host weight writes after initialization. Host DDR stores the records; card SRAM holds control buffers.',31,GREEN)
  text(d,(135,2041),'Recorded hardware replay · logical memory layout · motion is illustrative · no CPU or energy advantage claimed',29,MUTED)
  text(d,(135,2093),'github.com/MidasMulli/akd1500-lab',28,BLUE)
  bases.append(im)
 def frame(t):
  stage=min(3,int(t/4));im=bases[stage].copy();d=ImageDraw.Draw(im);progress=(t%4)/4
  # Motion shows logical dispatch/update flow only; no invented spike timing.
  if stage in [1,2]:
   side=0 if stage==1 else 1;x=[130,1980][side];gx,gy=x+255,365+278;idx=wrong if side==0 else target;end=(gx+(idx%128)*10+5,gy+(idx//128)*9+4);start=(x+255,365+1130)
   d.line((start,end),fill='#65839c',width=3);q=(.5-.5*math.cos(progress*math.pi));cx=start[0]+(end[0]-start[0])*q;cy=start[1]+(end[1]-start[1])*q;col=RED if side==0 else GREEN
   d.ellipse((cx-12,cy-12,cx+12,cy+12),fill=col)
  d.rectangle((2550,2107,3700,2114),fill='#20354b');d.rectangle((2550,2107,2550+int(1150*min(t/16,1)),2114),fill=BLUE)
  return im
 out=P/'visual/fnp2-window';poster=frame(14);poster.save(out.with_suffix('.png'));frame(1).resize((1920,1080),Image.Resampling.LANCZOS).save(P/'visual/fnp2-window-opening.png')
 if video:
  path=P/'visual/fnp2-window-4k60.mp4';cmd=['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','60','-i','-','-an','-c:v','libx264','-preset','veryfast','-crf','17','-threads','6','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
  proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  try:
   for i in range(960):proc.stdin.write(frame(i/60).tobytes())
   proc.stdin.close();assert proc.wait()==0
  except Exception:proc.kill();raise
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(path),'-vf','scale=1920:1080','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(P/'visual/fnp2-window-1080p60.mp4')],check=True)
 outputs=[out.with_suffix('.png'),P/'visual/fnp2-window-opening.png']
 if video:outputs += [P/'visual/fnp2-window-4k60.mp4',P/'visual/fnp2-window-1080p60.mp4']
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 binding=dict(status='RECORDED_RAM_AND_RESPONSE_REPLAY_NOT_ELECTRICAL_TRACE',archive_sha256=sha(BASE.with_suffix('.npz')),metadata_sha256=sha(BASE.with_suffix('.json')),renderer_sha256=sha(Path(__file__)),selected_seed=107101,selected_global_record=target,actual_direct_changed_record=wrong,window_local_record=local,removed_connection_bit=removed,added_connection_bit=added,main_sealed_updates=summary['main_sealed_updates'],outputs={p.name:sha(p) for p in outputs},limits=m['limitations'])
 (P/'visual/FNP2_WINDOW_SOURCE_BINDING.json').write_text(json.dumps(binding,indent=2)+'\n')
 print(json.dumps(dict(status='VISUAL_RENDERED',outputs=[str(p) for p in outputs])))
if __name__=='__main__':main('--video' in sys.argv)
