// Deterministic editorial capture of this Explorer's measured scene and records.
import * as THREE from './vendor/three.module.js';
import {matchingKey} from './repeated_memory.js';
export async function installRepeatedFilm(a){
 const D=a.data,E=D.arms.native.events,C=document.createElement('canvas'),g=C.getContext('2d',{alpha:false}),R=a.renderer;
 const W=1920,H=1080,rect={x:330,y:265,w:1010,h:625},bg='#08111b',white='#e4edf4',dim='#91a5b8',cyan='#8bd9e5',gold='#efbd87';
 const images=new Map();await Promise.all(Object.entries(D.images).map(async([id,v])=>{const im=new Image();im.src=v.data_url;await im.decode();images.set(id,im);}));
 const schedule=[];function add(e,duration,section){schedule.push({source:'native',index:e.index,duration,section});}
 for(let key=0;key<8;key++){const seq=E.filter(e=>e.role==='fill'&&e.key===key&&e.active_bank===1);for(const e of seq.filter(e=>e.changes.some(c=>c.bank===1)).slice(0,2))add(e,.12,'Fill the eight native records');add(seq.at(-1),.13,'Fill the eight native records');}
 add(E.filter(e=>e.role==='before'&&e.active_bank===1).at(-1),1.2,'All eight stored descriptors are recalled');
 for(let cy=0;cy<4;cy++){
  const es=E.filter(e=>e.cycle===cy),rev=es.filter(e=>e.role==='reverse'&&e.changes.some(c=>c.bank===0));for(const e of rev.slice(0,2))add(e,.4,'The learned usefulness rule changes');
  const ret=es.filter(e=>e.role==='retire');add(ret.find(e=>e.active_bank===0&&e.control_flags[3]),.4,'Native recall and policy permit retirement');add(ret.find(e=>e.changes.some(c=>c.bank===1)),.6,'The matching record becomes reusable');
  const ns=es.filter(e=>e.role==='new_key'&&e.active_bank===1);for(const e of ns.filter(e=>e.changes.some(c=>c.bank===1)).slice(0,3))add(e,.3,'A new image descriptor takes that record');add(ns.at(-1),.35,'The replacement is now recalled');
  const delay=es.filter(e=>e.role==='delay');for(let j=0;j<16;j++){const seq=delay.slice(j*12,(j+1)*12);add(seq[j%2?11:10],.09,'16 other image observations intervene');}
  const q=es.filter(e=>e.role==='delayed'&&e.active_bank===1);for(const e of q)add(e,.17,'Recall after the intervening observations');add(q.find(e=>e.key===8+cy),.75,'The new episode survives the delay');
 }
 add(E.findLast(e=>e.role==='delayed'&&e.key===11&&e.active_bank===1),2.5,'Four replacements · eight useful records retained');
 const starts=[];let total=0;for(const s of schedule){starts.push(total);total+=s.duration;}
 let scale=1,last=-1,center=new THREE.Vector3(),distance=14,direction=new THREE.Vector3(.35,.28,1).normalize();
 function setResolution(n){scale=n;C.width=W*n;C.height=H*n;g.setTransform(n,0,0,n,0,0);R.setPixelRatio(1);R.setSize(rect.w*n,rect.h*n,false);a.camera.aspect=rect.w/rect.h;a.camera.updateProjectionMatrix();}
 function restore(){const b=document.getElementById('sceneWrap').getBoundingClientRect();R.setPixelRatio(Math.min(devicePixelRatio,2));R.setSize(b.width,b.height,false);a.camera.aspect=b.width/b.height;a.camera.updateProjectionMatrix();}
 function text(v,x,y,size=18,color=white,weight=400){g.font=weight+' '+size+'px Arial,sans-serif';g.fillStyle=color;g.fillText(v,x,y);}
 function rule(x,y,w){g.strokeStyle='#263847';g.lineWidth=1;g.beginPath();g.moveTo(x,y);g.lineTo(x+w,y);g.stroke();}
 function draw(seconds){
  if(a.armTag()!=='native')throw Error('This deterministic film is bound to the primary native recording');
  seconds=Math.max(0,Math.min(total-.00001,seconds));let k=0;while(k+1<starts.length&&seconds>=starts[k+1])k++;
  const s=schedule[k],e=E[s.index];
  if(last!==s.index){
   document.querySelector('#followEdits').checked=false;
   const change=e.changes.find(c=>c.added.length||c.removed.length),bank=change?.bank??e.active_bank,record=change?.record??(bank===1?e.returned_record??0:e.native_responses[0]?.[0]?0:48);
   a.setSelection(bank,record);a.select(s.index);
   if(last<0){const box=new THREE.Box3().setFromObject(a.dynamic());box.getCenter(center);const size=box.getSize(new THREE.Vector3());distance=Math.max(size.y/2/Math.tan(Math.PI/9),size.x/2/Math.tan(Math.PI/9)/(rect.w/rect.h))*1.45;}
   last=s.index;
  }
  if(R.domElement.width!==rect.w*scale||R.domElement.height!==rect.h*scale){R.setPixelRatio(1);R.setSize(rect.w*scale,rect.h*scale,false);}a.camera.aspect=rect.w/rect.h;a.camera.updateProjectionMatrix();
  const dir=direction.clone().applyAxisAngle(new THREE.Vector3(0,1,0),.10*Math.sin(seconds*.17));a.camera.position.copy(center).addScaledVector(dir,distance);a.camera.lookAt(center);R.render(a.scene,a.camera);
  g.fillStyle=bg;g.fillRect(0,0,W,H);text('AKD1500 / NATIVE VISUAL MEMORY',48,55,22,cyan,600);text('A memory network that makes room',48,115,39,white,500);
  text((e.cycle>=0?'Cycle '+(e.cycle+1)+' / 4  ·  ':'')+s.section,48,170,23);text('RECORDED HARDWARE REPLAY',1510,55,14,gold,600);text('6 resident NPUs · 10 plastic units',1438,90,17,dim);rule(48,204,1824);
  text(e.image_id?'RECORDED IMAGE INPUT':'CONTEXT TEACHER',52,259,15,dim,600);
  if(e.image_id){g.drawImage(images.get(e.image_id),52,280,240,240);const im=D.images[e.image_id];text('Episode E'+e.key,52,556,25);text('CORe50 · object '+im.object,52,588,17,dim);text('Frozen host CNN → descriptor',52,618,16,dim);}
  else{const active=new Set(e.derived_features[0]);for(let j=0;j<39;j++){g.fillStyle=active.has(4*(64+j))?cyan:'#1c2d3d';g.fillRect(52+j%7*30,293+Math.floor(j/7)*30,24,24);}text('Context '+e.quality,52,532,25);text('Teach '+(e.teaching_label===0?'useful':'not useful'),52,565,19,dim);}
  rule(52,646,240);text('LAST NATIVE CONTROL',52,680,15,dim,600);['Recall feedback','Useful','Not useful','Retire permitted'].forEach((v,j)=>{text(v,52,718+j*34,17,dim);text(String(e.control_flags[j]),278,718+j*34,20,e.control_flags[j]?cyan:dim,600);});
  text('Captured event '+(e.index+1),52,906,16,dim);const edits=e.counts.reduce((n,c)=>n+c.added,0);text(edits?edits+' connection swaps':'Connections unchanged',52,935,17,edits?gold:dim);
  g.drawImage(R.domElement,rect.x,rect.y,rect.w,rect.h);
  for(const {el,position}of a.labels()){const p=position.clone().project(a.camera);if(p.z< -1||p.z>1)continue;g.textAlign='center';text(el.textContent,rect.x+(p.x+1)*rect.w/2,rect.y+(1-p.y)*rect.h/2,el.querySelector('strong')?20:13,el.style.color||dim,el.querySelector('strong')?600:400);}g.textAlign='left';
  text('STORED DESCRIPTORS / IMAGE REFERENCES',1385,259,14,dim,600);const state=a.stateAt(false)[1],response=e.native_responses[1];
  for(let j=0;j<8;j++){const x=1385+j%2*243,y=282+Math.floor(j/2)*143,rec=state[j],key=matchingKey(D,rec),edit=e.changes.some(c=>c.bank===1&&c.record===j),hot=response?.[j]===1;g.fillStyle=hot?'#152e3b':'#0d1b27';g.fillRect(x,y,226,130);g.strokeStyle=edit?'#7be1c0':hot?cyan:'#26394b';g.strokeRect(x+.5,y+.5,225,129);
   if(key>=0)g.drawImage(images.get('episode'+key),x+9,y+9,88,88);else{text(rec.words[3]>>>16===0?'Unused':rec.words[3]>>>16===1?'Reusable':'Updating',x+12,y+52,17,dim);}
   text('M'+j,x+110,y+33,17,dim,600);text(key>=0?'E'+key:'—',x+110,y+68,26,hot?cyan:white,500);text('Learn >'+(rec.words[3]>>>16),x+110,y+99,14,dim);
   const stored=new Set(rec.connections);for(let bit=0;bit<64;bit++){g.fillStyle=stored.has(4*bit)?edit?'#7be1c0':'#729aad':'#1b2d3b';g.fillRect(x+9+bit*3.22,y+111,2.2,6);}
  }
  text(response?'NATIVE CONTENT RETURN':'CONTENT RETURN PENDING',1385,894,15,dim,600);
  for(let j=0;j<8;j++){const x=1385+j*58;g.fillStyle=response?.[j]?'#8bd9e5':'#1a2e3c';g.fillRect(x,912,46,28);text(response?String(response[j]):'·',x+18,933,17,response?.[j]?bg:dim,600);}
  rule(48,974,1824);const stats=D.arms.native.assessment;text('Neuron cores: returned bits · input pins: captured inputs · green/red: connection edits',48,1007,17,dim);text('Card-side feedback · host-paced roles · logical geometry',48,1043,16,dim);
  text(stats.new_recall.join('/')+' new episodes',1385,1007,20,cyan,500);text(stats.delayed_recall.join('/')+' delayed recalls',1385,1043,20,cyan,500);
  return {source:'native',index:s.index,section:s.section,cycle:e.cycle};
 }

const encode=async({fps=60}={})=>{a.stop();setResolution(2);const n=Math.ceil(total*fps),chunks=[],manifest={frames:n,width:3840,height:2160,fps,source_seconds:total,timing:'Deterministiceditorialreplay,notnativeeventlatency',native_memory_return_bits:8,native_policy_flags:2,displayed_plastic_units:10,interpolation:false,events:[],error:null};let failed=null;const config={codec:'avc1.640034',width:3840,height:2160,bitrate:35000000,framerate:fps,hardwareAcceleration:'prefer-hardware',latencyMode:'realtime',avc:{format:'annexb'}};const support=await VideoEncoder.isConfigSupported(config);if(!support.supported)throw Error('Requested H2644K60 configuration unavailable');const encoder=new VideoEncoder({output:chunk=>{const a=new Uint8Array(chunk.byteLength);chunk.copyTo(a);chunks.push(a);},error:e=>failed=e});encoder.configure(config);let key='';for(let i=0;i<n;i++){const e=draw(i/fps),next=e.source+':'+e.index;if(next!==key){manifest.events.push({frame:i,...e});key=next;}const f=new VideoFrame(C,{timestamp:Math.round(i*1e6/fps),duration:Math.round(1e6/fps)});encoder.encode(f,{keyFrame:i%(fps*2)===0});f.close();while(encoder.encodeQueueSize>6){if(failed)throw failed;await new Promise(r=>setTimeout(r,1));}window.renderProgress={frame:i+1,total:n};if(i%30===0)await new Promise(r=>setTimeout(r,0));}await encoder.flush();encoder.close();if(failed)throw failed;const blob=new Blob(chunks,{type:'video/h264'}),downloadAnchor=document.createElement('a');downloadAnchor.href=URL.createObjectURL(blob);downloadAnchor.download='AKD_NATIVE_MEMORY_4K60.h264';downloadAnchor.click();manifest.bytes=blob.size;window.renderManifest=manifest;window.renderProgress={frame:n,total:n,done:true};restore();return manifest;};
window.contentFilm={ready:true,duration:total,draw,canvas:C,encode,poster:(t=total-.1)=>{setResolution(2);draw(t);const png=C.toDataURL('image/png');restore();return png;},preflight:async()=>({webgl:R.getContext().getParameter(R.getContext().VERSION),webcodecs:typeof VideoEncoder!=='undefined',fourK:await VideoEncoder.isConfigSupported({codec:'avc1.640034',width:3840,height:2160,bitrate:35000000,framerate:60,hardwareAcceleration:'prefer-hardware',latencyMode:'realtime',avc:{format:'annexb'}})})};
window.contentFilm.encode=(original=>async(options={})=>{setResolution(2);last=-1;return original(options);})(encode);
}
