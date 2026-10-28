import * as T from './vendor/three.module.js';
const D=await(await fetch('./data.json')).json();
const C=document.querySelector('#film'),g=C.getContext('2d',{alpha:false});
const W=1920,H=1080,box={x:300,y:245,w:1035,h:700};
const col={ink:'#071019',paper:'#e4eef5',dim:'#90a5b7',cyan:'#91dce8',green:'#75e1bb',orange:'#f2af77',purple:'#b6a4ee'};
const R=new T.WebGLRenderer({alpha:true,antialias:true,preserveDrawingBuffer:true}),scene=new T.Scene(),camera=new T.PerspectiveCamera(34,box.w/box.h,.1,100);
R.setPixelRatio(1);R.setSize(box.w,box.h);R.setClearColor(0,0);
scene.add(new T.AmbientLight(0xbcd7ee,.9));const light=new T.DirectionalLight(0xffffff,1.6);light.position.set(3,4,6);scene.add(light);
const nodes=[],cores=[],halos=[],inputPositions=[],memoryPositions=[],lines=new T.Group();scene.add(lines);
for(let j=0;j<176;j++){
  const a=j/176*Math.PI*2,z=((j*47)%176)/176*2-1,p=new T.Vector3(-1.65+Math.cos(a)*1.5,Math.sin(a)*2.5,z*.9);
  const m=new T.Mesh(new T.SphereGeometry(.046,12,8),new T.MeshBasicMaterial({color:0x294354,transparent:true,opacity:.35}));
  m.position.copy(p);scene.add(m);nodes.push(m);inputPositions.push(p);
}
for(let j=0;j<32;j++){
  const p=new T.Vector3(1.22+(j%4)*.57,2.62-Math.floor(j/4)*.74,.6*Math.sin(j*1.7));
  const m=new T.Mesh(new T.SphereGeometry(.084,20,14),new T.MeshStandardMaterial({color:0x407487,emissive:0x102835,roughness:.25,metalness:.22}));
  m.position.copy(p);scene.add(m);cores.push(m);memoryPositions.push(p);
  const h=new T.Mesh(new T.SphereGeometry(.17,16,12),new T.MeshBasicMaterial({color:0x88dce8,transparent:true,opacity:.025,depthWrite:false}));
  h.position.copy(p);scene.add(h);halos.push(h);
}
let scale=1,playing=true,clock=0,lastTime=performance.now(),last=-1,yaw=0,pitch=0,drag=null;
const pictures={};await Promise.all(Object.entries(D.images).map(async([k,v])=>{const im=new Image();im.src=v.src;await im.decode();pictures[k]=im;}));
const schedule=[],starts=[];let total=0;
for(let j=0;j<D.events.length;j++){
  const e=D.events[j],featured=e.tag.startsWith('S2c00_');
  const duration=e.role==='query'?(e.response.some(Boolean)?(featured?1.1:.10):e.candidate===2?(featured?.6:.055):.018):e.changes.length?(featured?.18:.045):.018;
  schedule.push({j,duration});
}
const base=schedule.reduce((a,s)=>a+s.duration,0),factor=Math.max(1,43.5/base);
for(const s of schedule){starts.push(total);s.duration*=factor;total+=s.duration;}
starts.push(total);schedule.push({j:D.events.length-1,duration:1.5});total+=1.5;
function select(e){
  for(const x of lines.children.slice()){lines.remove(x);x.geometry.dispose();x.material.dispose();}
  const active=new Set(e.active),memory=D.memory_versions[e.memory_id],buckets={dim:[],active:[],added:[],removed:[]};
  nodes.forEach((m,j)=>{const on=active.has(j*4);m.material.color.set(on?0xa2e8f3:0x294354);m.material.opacity=on?1:.16;m.scale.setScalar(on?1.4:1);});
  for(let j=0;j<32;j++){
    const edit=e.changes.find(c=>c.record===j),addresses=[...new Set([...memory[j].addresses,...(edit?.removed||[])])];
    for(const a of addresses){
      const p=inputPositions[a/4],q=memoryPositions[j];if(!p)throw Error('Nonexistent recorded channel '+a);
      const bucket=edit?.added.includes(a)?'added':edit?.removed.includes(a)?'removed':active.has(a)?'active':'dim';
      const bend=new T.Vector3((p.x+q.x)/2,(p.y+q.y)/2,.8+((j%4)-1.5)*.2),curve=new T.QuadraticBezierCurve3(p,bend,q),points=curve.getPoints(12);
      for(let k=0;k<12;k++)buckets[bucket].push(...points[k].toArray(),...points[k+1].toArray());
    }
    const hot=e.response[j],changed=!!edit;
    cores[j].material.color.set(changed?0x75e1bb:hot?0xb2f2ff:0x245366);
    cores[j].material.emissive.set(changed?0x206c55:hot?0x498fa5:0x0b202a);
    cores[j].scale.setScalar(hot?1.24:changed?1.12:1);
    halos[j].material.color.set(changed?0x75e1bb:0x91dce8);halos[j].material.opacity=changed?.32:hot?.38:.025;
  }
  for(const [name,vertices]of Object.entries(buckets)){
    if(!vertices.length)continue;
    const geom=new T.BufferGeometry();geom.setAttribute('position',new T.Float32BufferAttribute(vertices,3));
    const material=new T.LineBasicMaterial({color:{dim:0x37596d,active:0x80d4e5,added:0x75e1bb,removed:0xe7a189}[name],transparent:true,opacity:{dim:.04,active:.36,added:.95,removed:.85}[name],depthWrite:false});
    lines.add(new T.LineSegments(geom,material));
  }
}
const label=['No event','Goal','Contact'];
function text(s,x,y,n=18,color=col.paper,weight=400){g.font=weight+' '+n+'px Arial,sans-serif';g.fillStyle=color;g.fillText(s,x,y);}
function rule(x,y,w){g.strokeStyle='#253848';g.lineWidth=1;g.beginPath();g.moveTo(x,y);g.lineTo(x+w,y);g.stroke();}
function round(x,y,w,h,color,r=8){g.fillStyle=color;g.beginPath();g.roundRect(x,y,w,h,r);g.fill();}
function bars(p,x,y,w){for(let j=0;j<3;j++){text(label[j],x,y+j*29,15,col.dim);round(x+88,y-12+j*29,w-148,11,'#172c3a',3);round(x+88,y-12+j*29,Math.max(1,(w-148)*p[j]),11,j===2?col.orange:j===1?col.cyan:col.purple,3);text(Math.round(p[j]*100)+'%',x+w-44,y+j*29,15,col.paper,600);}}
function draw(seconds){
  seconds=Math.max(0,Math.min(total-.00001,seconds));let k=0;while(k+1<starts.length&&seconds>=starts[k+1])k++;
  const e=D.events[schedule[k].j],memory=D.memory_versions[e.memory_id];
  if(last!==e.index){select(e);last=e.index;}
  const theta=.20*Math.sin(seconds*.105)+yaw;camera.position.set(Math.sin(theta)*11.5,.65+pitch,Math.cos(theta)*11.5);camera.lookAt(.1,0,0);R.render(scene,camera);
  g.setTransform(scale,0,0,scale,0,0);g.fillStyle=col.ink;g.fillRect(0,0,W,H);
  text('AKD1500 × WORLD MODEL',44,48,19,col.cyan,600);
  text('The model stays fixed. The chip learns what to keep.',44,105,37,col.paper,500);
  text('32 resident memory units · native learning across 7 mapped processors',46,148,21,col.dim);
  text('MEASURED CHIP REPLAY',1634,48,15,col.orange,600);text('144 causal episodes',1680,83,17,col.dim);rule(44,177,1832);
  const phases=['First situations · original physics','First situations · changed physics','New situations · original physics','New situations · changed physics','Earlier situations return · original physics','Earlier situations return · changed physics'];
  text(phases[e.phase],44,217,23);round(1440,191,435,39,'#102b37');text(e.role==='query'?'READ MEMORY · before outcome':'LEARN / RETAIN · outcome received',1455,217,16,col.cyan,600);
  for(const [kind,title,y]of [['observed_t5','OBSERVED INPUT · t5',265],['imagined_t25','FROZEN MODEL · IMAGINED t25',527],['true_final','ACTUAL OUTCOME',789]]){
    text(title,44,y-10,13,col.dim,600);
    if(kind!=='true_final'||e.role==='feedback'){g.drawImage(pictures[e.tag+'_'+kind],44,y,218,218);g.strokeStyle='#273d4d';g.strokeRect(44.5,y+.5,217,217);}
    else{round(44,y,218,218,'#0c1e2b');text('Outcome arrives',67,y+92,18,col.dim);text('after the query',67,y+120,18,col.dim);}
  }
  g.drawImage(R.domElement,box.x,box.y,box.w,box.h);
  text('CAPTURED INPUT → NATIVE CONNECTIONS → MEASURED RETURNS',332,277,14,col.dim,600);
  text('176 sparse channels',340,934,17,col.dim);text('32 units · one memory layer [3,2,0]',915,934,16,col.dim);
  function project(p){const a=p.clone().project(camera);return{x:box.x+(a.x+1)*box.w/2,y:box.y+(1-a.y)*box.h/2};}
  for(let j=0;j<32;j++){const p=project(memoryPositions[j]);text('M'+j,p.x+11,p.y+5,12,e.response[j]?col.cyan:col.dim,600);}
  const rx=1382,ready=e.role!=='query'||e.candidate===2,known=e.query_value>=0;
  text(e.tag.replaceAll('_',' · '),rx,272,15,col.dim,600);text('Frozen world model',rx,307,21);bars(e.wm,rx,338,487);
  text('World model + native memory',rx,456,21,col.cyan);
  if(ready){bars(e.combined,rx,487,487);text(known?'Recalled outcome: '+label[e.query_value]:'No recall · use model forecast',rx,588,17,known?col.cyan:col.dim);}
  else text('Querying candidate '+(e.candidate+1)+' / 3',rx,505,19,col.dim);
  rule(rx,613,488);const used=memory.filter(m=>m.owner!==null).length;
  text('ON-CHIP MEMORY',rx,642,15,col.dim,600);text(used+' / 32 stored',1710,642,15,col.cyan,600);
  for(let j=0;j<32;j++){
    const x=rx+(j%4)*124,y=658+Math.floor(j/4)*38,rec=memory[j],owner=rec.owner===null?null:D.catalog[rec.owner],edit=e.changes.find(c=>c.record===j);
    round(x,y,116,33,e.response[j]?'#174554':'#0e202e',4);
    if(edit){g.strokeStyle=col.green;g.strokeRect(x+.5,y+.5,115,32);}
    text('M'+j,x+6,y+12,10,col.dim,600);
    text(owner?owner.set+'.'+owner.context+' '+owner.regime+' '+['D','G','C'][owner.outcome]:'available',x+6,y+26,11,edit?col.green:col.paper);
    if(e.response[j]){g.fillStyle=col.cyan;g.beginPath();g.arc(x+106,y+10,3,0,Math.PI*2);g.fill();}
  }
  rule(rx,973,488);text('NEW SITUATIONS · correct second-encounter forecasts',rx,992,12,col.dim,600);
  for(const [j,name,title]of [[0,'error-only','Selective'],[1,'all-admit','Teach all'],[2,'S1AB-frozen','Frozen']]){
    const v=e.progress[name];text(title+' '+v.correct+'/'+v.n,rx+j*164,1014,16,j===0?col.cyan:col.dim,600);
  }
  rule(44,1029,1832);const changed=e.changes.reduce((n,c)=>n+c.added.length,0),hot=e.response.map((v,j)=>v?'M'+j:null).filter(Boolean);
  text('Native forward '+e.index+' / '+D.source.source_forwards,44,1062,15,col.dim);
  text(changed?changed+' measured connection '+(changed===1?'replacement':'replacements'):'Connections retained',358,1062,16,changed?col.green:col.dim);
  text('Returned: '+(hot.join(', ')||'none'),823,1062,16,hot.length?col.cyan:col.dim);
  text('Recorded chip · reconstructed scenes · logical layout',1400,1062,14,col.dim);
  document.querySelector('#seek').value=String(Math.round(seconds/total*1000));document.querySelector('#time').textContent=seconds.toFixed(1)+' / '+total.toFixed(1)+'s';
  return{source:'native',index:e.index,episode:e.episode,phase:e.phase,tag:e.tag,ram_edits:e.changes.length,query:e.query_value,progress:e.progress};
}
function resolution(n){scale=n;C.width=W*n;C.height=H*n;R.setSize(box.w*n,box.h*n,false);}
document.querySelector('#play').onclick=()=>{playing=!playing;document.querySelector('#play').textContent=playing?'Pause':'Play';};
document.querySelector('#seek').oninput=e=>{clock=Number(e.target.value)/1000*total;draw(clock);};
C.onpointerdown=e=>{drag=[e.clientX,e.clientY,yaw,pitch];C.setPointerCapture(e.pointerId);};
C.onpointermove=e=>{if(drag){yaw=drag[2]+(e.clientX-drag[0])*.003;pitch=Math.max(-2,Math.min(2,drag[3]+(e.clientY-drag[1])*.006));draw(clock);}};C.onpointerup=()=>drag=null;
for(const v of D.limits){const li=document.createElement('li');li.textContent=v;document.querySelector('#limits').append(li);}
document.querySelector('#source').textContent='Native source '+D.source.native_archive+' · SHA256 '+D.source.native_sha256;
function animate(now){if(playing){clock=(clock+Math.min(.1,(now-lastTime)/1000))%total;draw(clock);}lastTime=now;requestAnimationFrame(animate);}requestAnimationFrame(animate);
async function encode({fps=60}={}){
  playing=false;yaw=pitch=0;resolution(2);const n=Math.ceil(total*fps),chunks=[],manifest={frames:n,width:3840,height:2160,fps,source_seconds:total,timing:'Editorial replay timing, not latency or electrical spike timestamps',events:[],source:D.source};
  let failed=null;const config={codec:'avc1.640034',width:3840,height:2160,bitrate:35000000,framerate:fps,hardwareAcceleration:'prefer-hardware',latencyMode:'realtime',avc:{format:'annexb'}};
  if(!(await VideoEncoder.isConfigSupported(config)).supported)throw Error('4K60 encoder unavailable');
  const en=new VideoEncoder({output:c=>{const a=new Uint8Array(c.byteLength);c.copyTo(a);chunks.push(a);},error:e=>failed=e});en.configure(config);
  let previous=-1;
  for(let i=0;i<n;i++){
    const e=draw(i/fps);if(e.index!==previous){manifest.events.push({frame:i,...e});previous=e.index;}
    const f=new VideoFrame(C,{timestamp:Math.round(i*1e6/fps),duration:Math.round(1e6/fps)});en.encode(f,{keyFrame:i%(fps*2)===0});f.close();
    while(en.encodeQueueSize>6){if(failed)throw failed;await new Promise(r=>setTimeout(r,1));}
    window.renderProgress={frame:i+1,total:n};if(i%30===0)await new Promise(r=>setTimeout(r,0));
  }
  await en.flush();en.close();if(failed)throw failed;const blob=new Blob(chunks,{type:'video/h264'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='AKD_NATIVE_MEMORY_4K60.h264';a.click();manifest.bytes=blob.size;window.renderProgress={frame:n,total:n,done:true};resolution(1);return manifest;
}
window.contentFilm={ready:true,duration:total,draw,canvas:C,encode,poster:(t=total-.1)=>{playing=false;resolution(2);draw(t);const p=C.toDataURL('image/png');resolution(1);return p;},preflight:async()=>({source:D.source,visual_events:D.events.length,units:32,webgl:R.getContext().getParameter(R.getContext().VERSION),webcodecs:typeof VideoEncoder!=='undefined',duration:total})};draw(0);
