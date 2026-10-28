import * as THREE from '../vendor/three.module.js';
import {filmData as D} from './session.js';
const C=document.getElementById('film'),g=C.getContext('2d'),S=D.session,E=S.events,B=D.banks;
const palette=['#6eb8ff','#e5bf76','#bc9df4'],names=['SOURCE CHECKER','MEMORY GROUP A','MEMORY GROUP B'];
const rect={x:40,y:208,w:1256,h:660};
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true,preserveDrawingBuffer:true});renderer.setSize(rect.w*1.5,rect.h*1.5,false);renderer.setClearColor('#080e17',1);renderer.outputEncoding=THREE.sRGBEncoding;
const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(26,rect.w/rect.h,.1,100);
scene.add(new THREE.AmbientLight('#c3d9ec',.65));const light=new THREE.DirectionalLight('#ddebff',1.8);light.position.set(2,5,9);scene.add(light);
let group=new THREE.Group();scene.add(group);let nodeObjects=[],flow=[],selected={bank:0,record:39},lastEvent=-1;
const sphere=new THREE.IcosahedronGeometry(.115,2),dot=new THREE.SphereGeometry(.033,7,6);
const states=[];let current=S.initial.map(b=>b.slice());
for(const e of E){const next=current.map(b=>b.slice());for(const c of e.changes)next[c.bank][c.record]=c.after;states.push(next);current=next;}
const duration=i=>i<84?.105:i===84?2:i<91?.5:i<93?1.65:i<97?.22:i===97?1.3:i===98?2:i===99?1:i<103?.4:3;
const starts=[];let total=0;for(let i=0;i<E.length;i++){starts.push(total);total+=duration(i);}const filmDuration=total;
function bx(k){return (k-1)*4.35;}
function channel(k,id){const a=id*2.39996323,r=.43+Math.sqrt((id%109)/109)*1.12;return new THREE.Vector3(bx(k)+Math.cos(a)*r,2.05+Math.sin(a)*r*.88,Math.sin(id*.713)*.6);}
function unit(k,j){const n=B[k].plastic.indexOf(j);return new THREE.Vector3(bx(k)+((n%4)-1.5)*.52,-.85-Math.floor(n/4)*.45,Math.sin(n*1.9)*.17);}
function dispose(){scene.remove(group);group.traverse(o=>{if(o.geometry&&!o.userData.shared)o.geometry.dispose();if(o.material)o.material.dispose();});group=new THREE.Group();scene.add(group);nodeObjects=[];flow=[];}
function segment(a,b,c,opacity){const mid=a.clone().lerp(b,.5);mid.z+=.18;const curve=new THREE.QuadraticBezierCurve3(a,mid,b);const geo=new THREE.BufferGeometry().setFromPoints(curve.getPoints(10));const line=new THREE.Line(geo,new THREE.LineBasicMaterial({color:c,transparent:true,opacity,depthWrite:false}));group.add(line);return curve;}
function choose(i){const e=E[i],edits=e.changes.filter(c=>c.added.length||c.removed.length).sort((a,b)=>b.added.length-a.added.length);if(edits.length){selected={bank:edits[0].bank,record:edits[0].record};return;}if(e.role==='exam'||e.role==='query'){for(const k of [2,1,0]){const j=B[k].plastic.find(j=>e.derived_potentials[k][j]>B[k].thresholds[j]);if(j!==undefined){selected={bank:k,record:j};return;}}}}
function rebuild(i){dispose();choose(i);const e=E[i],state=states[i];for(let k=0;k<B.length;k++){
 const active=new Set(e.derived_features[k]),ids=new Set();for(const j of B[k].plastic)for(const id of state[k][j].connections)ids.add(id);
 const changes=new Map(e.changes.filter(c=>c.bank===k).map(c=>[c.record,c]));for(const c of changes.values())for(const id of c.removed)ids.add(id);
 for(const id of ids){const mesh=new THREE.Mesh(dot,new THREE.MeshBasicMaterial({color:active.has(id)?palette[k]:'#314354',transparent:true,opacity:active.has(id)?.9:.36}));mesh.position.copy(channel(k,id));mesh.userData.shared=true;group.add(mesh);}
 for(const j of B[k].plastic){const pos=unit(k,j),emitted=e.derived_potentials[k][j]>B[k].thresholds[j],change=changes.get(j),sel=k===selected.bank&&j===selected.record;
  const material=new THREE.MeshStandardMaterial({color:palette[k],metalness:.32,roughness:.25,emissive:emitted?palette[k]:'#000000',emissiveIntensity:emitted?.75:0,transparent:true,opacity:emitted?1:.65});const mesh=new THREE.Mesh(sphere,material);mesh.position.copy(pos);mesh.scale.setScalar(sel?1.35:1);mesh.userData.shared=true;group.add(mesh);nodeObjects.push({mesh,emitted,k,j,change});
  if(sel||change?.added.length){const ring=new THREE.Mesh(new THREE.TorusGeometry(sel?.21:.17,.009,5,36),new THREE.MeshBasicMaterial({color:change?.added.length?'#7be1ba':'#eaf5ff',transparent:true,opacity:.9}));ring.position.copy(pos);group.add(ring);}
  for(const id of state[k][j].connections){const add=change?.added.includes(id),curve=segment(channel(k,id),pos,add?'#7be1ba':palette[k],add?.92:sel?.42:active.has(id)?.2:.055);
   if(active.has(id)){const p=new THREE.Mesh(dot,new THREE.MeshBasicMaterial({color:palette[k],transparent:true,opacity:.95}));p.userData.shared=true;group.add(p);flow.push({mesh:p,curve,offset:(id%5)*.03});}
  }
  for(const id of change?.removed||[])if(!state[k][j].connections.includes(id))segment(channel(k,id),pos,'#f18a92',.8);
 }
}lastEvent=i;}
function text(t,x,y,size=20,c='#e6edf5',weight=400){g.fillStyle=c;g.font=`${weight} ${size}px Arial,sans-serif`;g.fillText(t,x,y);}
function line(x,y,w,c='#26384a'){g.strokeStyle=c;g.lineWidth=1;g.beginPath();g.moveTo(x,y);g.lineTo(x+w,y);g.stroke();}
function panel(x,y,w,h){g.fillStyle='#0d1724';g.fillRect(x,y,w,h);g.strokeStyle='#223449';g.strokeRect(x+.5,y+.5,w-1,h-1);}
function phase(i){if(i<84)return ['WATCH THE NETWORK FORM MEMORIES','Recorded initial teaching, queries and connection changes'];if(i===84)return ['ASK WITHOUT TEACHING','The original answer is 0. All learning is disabled for this query.'];if(i<91)return ['TRY A CHANGE — IT IS REJECTED','The source checker blocks this proposal; memory connections stay unchanged.'];if(i<97)return ['GIVE FEEDBACK TO THE SOURCE CHECKER','Only its measured connection changes are highlighted.'];if(i===97)return ['TRY THE SAME PROPOSAL AGAIN','The returned permission has changed from reject to accept.'];if(i===98)return ['FOLLOW THE UPDATE INTO MEMORY','Native memory connections change after the accepted proposal.'];return ['ASK THE SAME QUESTION AGAIN','The measured answer is now 1. Querying does not teach it.'];}
function project(p){const q=p.clone().project(camera);return {x:rect.x+(q.x+1)*rect.w/2,y:rect.y+(1-q.y)*rect.h/2};}
function latest(i,kind){for(let j=i;j>=0;j--){const e=E[j];if(kind==='gate'&&e.role==='query'&&e.policy_key===1&&e.key===0&&e.policy_flags.some(Boolean))return {v:e.policy_flags[1]?'ACCEPT':'REJECT',event:j+1};if(kind==='answer'&&e.key===0&&['exam','query'].includes(e.role)&&e.memory_flags.some(Boolean))return {v:e.memory_flags[1]?'1':'0',event:j+1};}return null;}
function board(ids,oldSet,delta,x,y,title,before){text(title,x,y,15,'#91a5b8',600);for(let n=0;n<32;n++){const id=ids[n],isChange=before?delta?.removed.includes(id):delta?.added.includes(id),xx=x+(n%8)*60,yy=y+19+Math.floor(n/8)*35;g.fillStyle=isChange?(before?'#45222d':'#123d31'):'#162537';g.fillRect(xx,yy,54,29);text(String(id).padStart(3,'0'),xx+10,yy+20,15,isChange?(before?'#ffacb5':'#96f3c8'):'#a5b8cc');}}
function right(i){const e=E[i],k=selected.bank,j=selected.record,after=states[i][k][j],before=i?states[i-1][k][j]:S.initial[k][j],delta=e.changes.find(c=>c.bank===k&&c.record===j);const x=1360;
 text('INSPECT A LEARNING RECORD',x,138,16,'#91a5b8',600);text(`${B[k].name} / ${String(j).padStart(2,'0')}`,x,181,29,palette[k],500);text(`NPU (${B[k].mapping.join(', ')})  ·  captured RAM`,x,212,17,'#a5b8cc');
 text('ACTIVE INPUTS · RECONSTRUCTED',x,258,15,'#91a5b8',600);const active=new Set(e.derived_features[k]);for(let id=0;id<512;id++){g.fillStyle=active.has(id)?palette[k]:'#162537';g.fillRect(x+(id%64)*7.5,275+Math.floor(id/64)*7.5,5.5,5.5);}
 board(before.connections,null,delta,x,374,'BEFORE THIS EVENT',true);board(after.connections,null,delta,x,565,'AFTER THIS EVENT',false);
 text(`+${delta?.added.length||0} connected`,x,782,20,'#7be1ba');text(`−${delta?.removed.length||0} disconnected`,x+240,782,20,'#f18a92');
 text('MEASURED CONNECTION EDITS / EVENT',x,832,15,'#91a5b8',600);line(x,900,480);
 for(let k=0;k<3;k++){g.strokeStyle=palette[k];g.lineWidth=1.6;g.beginPath();for(let n=0;n<=i;n++){const xx=x+n/(E.length-1)*480,yy=897-E[n].counts[k].added*5;n?g.lineTo(xx,yy):g.moveTo(xx,yy);}g.stroke();}
 g.strokeStyle='#d4e6f5';const cx=x+i/(E.length-1)*480;g.beginPath();g.moveTo(cx,850);g.lineTo(cx,905);g.stroke();
}
function draw(seconds){let i=0;for(let n=0;n<starts.length;n++){if(seconds>=starts[n])i=n;else break;}const local=(seconds-starts[i])/duration(i),e=E[i];if(i!==lastEvent)rebuild(i);
 camera.position.set(Math.sin(seconds*.065)*.75,1.25+Math.sin(seconds*.07)*.12,15.5);camera.lookAt(0,.45,0);
 for(const p of flow){const t=Math.min(1,Math.max(0,local*1.7-p.offset));p.mesh.position.copy(p.curve.getPoint(t));p.mesh.visible=local<.85;}
 renderer.render(scene,camera);g.fillStyle='#080e17';g.fillRect(0,0,1920,1080);text('AKD1500 / NETWORK EXPLORER',44,57,31,'#eaf2fa',500);text('Inspect what learning changes',44,94,21,'#a5b8cc');text('RECORDED CHIP SESSION · CONDENSED PLAYBACK',1355,55,15,'#7be1ba',600);
 const [heading,detail]=phase(i);text(heading,44,151,27,'#e9f0f7',500);text(detail,44,184,19,'#a7b9ca');g.drawImage(renderer.domElement,rect.x,rect.y,rect.w,rect.h);
 for(let k=0;k<3;k++){const xx=rect.x+((k+.5)/3)*rect.w;text(names[k],xx-122,245,18,palette[k],600);text(`${B[k].plastic.length} learning records`,xx-122,273,15,'#859bb1');text(`NPU ${B[k].mapping.join('.')}`,xx-122,294,14,'#748aa0');}
 for(const n of nodeObjects){if(n.change?.added.length||n.k===selected.bank&&n.j===selected.record){const p=project(n.mesh.position);text(String(n.j).padStart(2,'0'),p.x-9,p.y+27,13,'#e0eaf5');}}
 right(i);line(44,921,1252);const gate=latest(i,'gate'),answer=latest(i,'answer');text(`EVENT ${String(i+1).padStart(3,'0')} / 104`,44,961,18,'#91a5b8',600);text(`Learning gate: ${gate?.v||'initial teaching'}`,300,961,22,gate?.v==='ACCEPT'?'#7be1ba':gate?.v==='REJECT'?'#f5c17b':'#a5b8cc');text(`Answer to fact 0: ${answer?.v||'not queried yet'}`,850,961,22,answer?.v==='1'?'#7be1ba':'#d5e3ee');
 let totals=[0,0,0];for(let n=0;n<=i;n++)for(let k=0;k<3;k++)totals[k]+=E[n].counts[k].added;text(`Captured connection replacements so far: ${totals.reduce((a,b)=>a+b,0)}   ·   this event: ${e.counts.map(c=>c.added).join(' / ')}`,44,1000,19,'#9aafc3');
 text('Measured: memory edits + returned answers. Reconstructed: input activity + glow. Logical layout; host-paced task. CPU controls also pass.',44,1047,17,'#8097ae');
 if(recorder?.state==='recording'){if(manifest.last_event!==i){manifest.rendered_events.push({event:i,recorded_second:seconds,selected:{...selected}});manifest.last_event=i;}manifest.rendered_frames++;}
 document.getElementById('state').textContent=`Event ${i+1}/104 · ${Math.min(seconds,filmDuration).toFixed(1)} / ${filmDuration.toFixed(1)}s · recorded replay`;
}
let running=false,start=0,recorder=null,chunks=[],manifest=null;
function frame(now){if(running){const t=(now-start)/1000;draw(Math.min(t,filmDuration));if(t>=filmDuration){running=false;if(recorder&&recorder.state==='recording')recorder.stop();document.getElementById('play').textContent='Replay walkthrough';}}requestAnimationFrame(frame);}
function play(){running=true;start=performance.now();lastEvent=-1;document.getElementById('play').textContent='Playing recorded session';}
function download(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
document.getElementById('play').onclick=play;
document.getElementById('record').onclick=()=>{if(recorder?.state==='recording')return;const mime=['video/webm;codecs=vp9','video/webm;codecs=vp8'].find(v=>MediaRecorder.isTypeSupported(v));if(!mime){document.getElementById('error').textContent='Browser recording codec unavailable';return;}chunks=[];const stream=C.captureStream(60);recorder=new MediaRecorder(stream,{mimeType:mime,videoBitsPerSecond:20000000});manifest={source_sha256:D.provenance.source_sha256,source:'Recorded104-event native session; condensed visual replay',native_new_sessions:0,duration_seconds:filmDuration,width:1920,height:1080,target_fps:60,rendered_frames:0,rendered_events:[],last_event:-1,timing:'Editorial event spacing; not hardware latency',evidence:D.provenance.evidence,events:E.map((e,i)=>({native_event:i,film_second:starts[i],duration:duration(i),counts:e.counts})),selected_record_rule:'Largest actual connection edit; otherwise a reconstructed responding plastic record'};recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};recorder.onstop=()=>{download(new Blob(chunks,{type:mime}),'AKD_EXPLORER_EXPLAINED_MASTER.webm');document.getElementById('manifest').disabled=false;stream.getTracks().forEach(t=>t.stop());document.getElementById('record').textContent='Export complete';};recorder.start(1000);play();};
window.addEventListener('error',e=>document.getElementById('error').textContent=e.message);draw(0);requestAnimationFrame(frame);

document.getElementById('manifest').onclick=()=>{if(manifest)download(new Blob([JSON.stringify(manifest,null,2)],{type:'application/json'}),'AKD_EXPLORER_EXPLAINED_CAPTURE.json');};
