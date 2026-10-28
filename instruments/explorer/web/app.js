import {installRepeatedFilm} from './repeated_film.js';
import {matchingKey,renderRepeatedComparison,renderRepeatedEvent} from './repeated_memory.js';
import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';
import {featurePosition,inputVolume,responseHalo,memorySetting} from './scene_atmosphere.js';
import {renderMemoryPanel,memorySummary} from './visual_memory.js';
import {neuronResponse,responseTrace} from './neuron_output.js';

const $=id=>document.getElementById(id), colors=['#6eb8ff','#e1bf7b','#c0a0ff','#7be1c0'];
let bankNames=['Manager','Memory 0','Memory 1'];
function bankX(k){return (k-((data?.banks.length||3)-1)/2)*4.25;}
function bankInputs(){return data.banks[bank].inputs||512;}
function bankBudget(){return data.banks[bank].num_weights||32;}
function isRepeated(){return data?.profile==='repeated-native-memory';}
function isRGB(){return data?.profile==='rgb-native-memory';}
function recordScore(k,j){return isRGB()?event().observed_potentials[k][j]:event().derived_potentials[k][j];}
let data,status,arm='native',index=0,bank=0,record=0,side='after',playing=false,lastTick=0;
let mode='replay',live=null,busy=false,allBanks=true,hoverChannel=null;
let backgroundOpacity=1,nodeView=false;
const caches=new Map(), error=e=>{$('error').textContent=String(e.message||e);$('error').hidden=false;};
if(window.akdDesktop){
  document.body.classList.add('desktop-host');$('desktopBar').hidden=false;
  const updateDesktop=s=>{
    $('desktopPin').textContent=s.alwaysOnTop?'Pinned':'Unpinned';$('desktopPin').setAttribute('aria-pressed',String(s.alwaysOnTop));
    $('desktopPass').disabled=!s.restoreRegistered;$('desktopPass').title=`Restore interaction with ${s.restoreShortcut}`;
    $('desktopPassHint').textContent=`Click-through · ${s.restoreShortcut} restores inspection`;$('desktopPassHint').hidden=!s.clickThrough;
  };
  const desktopCommand=action=>window.akdDesktop.command(action).then(updateDesktop).catch(error);
  window.akdDesktop.onState(updateDesktop);desktopCommand('state');
  $('desktopPin').onclick=()=>desktopCommand('pin');$('desktopPass').onclick=()=>desktopCommand('click-through');$('desktopClose').onclick=()=>desktopCommand('close');
}
function usingLive(){return mode==='live'&&live?.initial&&live?.events?.length>0;}
function currentArm(){return usingLive()?live:data.arms[arm];}
function events(){return currentArm()?.events||[];}
function event(){return events()[index];}
function cache(a){
  const key=usingLive()?'live':arm,old=caches.get(key);
  if(old&&old.length===a.events.length)return old;
  let state=a.initial.map(b=>b.slice()),frames=[];
  for(const e of a.events){
    let next=state.map(b=>b.slice());for(const c of e.changes)next[c.bank][c.record]=c.after;
    frames.push(next);state=next;
  }
  caches.set(key,frames);return frames;
}
function stateAt(before=false){const a=currentArm();if(!a?.initial)return null;const n=index-(before?1:0);return n<0?a.initial:cache(a)[n];}
function selectedChange(){return event()?.changes.find(c=>c.bank===bank&&c.record===record);}
function decode(f){return !f?.some(Boolean)?'No match':f.every(Boolean)?'Ambiguous':f[1]?'1':'0';}
function pretty(s){return s.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());}
function evidenceLabel(){
  const active=['ready','busy','starting','bootstrapping'].includes(live?.status);
  $('eventKind').textContent=usingLive()?(active?'RESIDENT TRIAL · CAPTURED EVENTS':'STOPPED TRIAL · RECORDED CAPTURES'):mode==='live'?'RECORDED PREVIEW · START A TRIAL FOR LIVE DATA':'RECORDED CHIP REPLAY';
  if(mode==='live')$('sourceBadge').textContent=usingLive()?(active?'NATIVE TRIAL · RESIDENT':'NATIVE TRIAL · STOPPED'):'RECORDED PREVIEW / LIVE CONTROLS';
  $('nodeEvidence').textContent=usingLive()?(active?'RESIDENT TRIAL':'RECORDED TRIAL'):'RECORDED REPLAY';
}
function playbackLabels(){$('play').textContent=playing?'Ⅱ':'▶';$('play').setAttribute('aria-label',playing?'Pause replay':'Play replay');$('nodePlay').textContent=playing?'Pause replay':'Play replay';}
function stopPlayback(){playing=false;playbackLabels();}

const scene=new THREE.Scene();scene.background=null;scene.fog=new THREE.FogExp2('#080e17',.022);
const camera=new THREE.PerspectiveCamera(38,1,.1,120);camera.position.set(10,8.5,17);
const renderer=new THREE.WebGLRenderer({canvas:$('scene'),antialias:true,alpha:true,preserveDrawingBuffer:true});
renderer.setClearColor('#080e17',1);
renderer.setPixelRatio(Math.min(devicePixelRatio,2));renderer.outputEncoding=THREE.sRGBEncoding;
const controls=new OrbitControls(camera,renderer.domElement);controls.target.set(0,.8,0);controls.enableDamping=true;controls.dampingFactor=.08;
controls.minDistance=7;controls.maxDistance=32;controls.maxPolarAngle=Math.PI*.78;controls.enablePan=true;
scene.add(new THREE.AmbientLight('#b5c6df',.6));const light=new THREE.DirectionalLight('#ffffff',1.4);light.position.set(4,8,9);scene.add(light);
const rim=new THREE.DirectionalLight('#6eb8ff',.8);rim.position.set(-8,3,-4);scene.add(rim);
const floor=new THREE.GridHelper(35,70,'#20334a','#122134');floor.position.y=-1.8;floor.material.transparent=true;floor.material.opacity=.28;scene.add(floor);
const setting=memorySetting();scene.add(setting);
let outputTrace=null;
let dynamic=new THREE.Group();scene.add(dynamic);let targets=[],labels=[];const raycaster=new THREE.Raycaster(),mouse=new THREE.Vector2();
const sphere=new THREE.IcosahedronGeometry(.105,3),inputGeometry=new THREE.SphereGeometry(.035,8,6);
function clearScene(){
  scene.remove(dynamic);dynamic.traverse(o=>{if(o.geometry&&!o.userData.shared)o.geometry.dispose();if(o.material)o.material.dispose();});
  dynamic=new THREE.Group();scene.add(dynamic);targets=[];labels=[];$('labels').replaceChildren();
}
function label(position,text,color,large=false){
  const el=document.createElement('div');el.className='sceneLabel';el.style.color=color;
  if(large){const strong=document.createElement('strong');strong.textContent=text;el.append(strong);}else el.textContent=text;
  $('labels').append(el);labels.push({el,position});
}
function neuronPosition(k,j,ids){
  if(isRGB()){
    const group=Math.floor(j/20),slot=j%20,a=group*Math.PI*2/11+Math.PI/2;
    const z=1-2*(slot+.5)/20,r=Math.sqrt(1-z*z),t=slot*2.399963229728653;
    return new THREE.Vector3(Math.cos(a)*3.35+Math.cos(t)*r*.45,.35+Math.sin(a)*1.65+Math.sin(t)*r*.34,Math.sin(group*1.7)*.57+z*.48);
  }
  const ix=ids.indexOf(j),cols=ids.length>20?8:4;
  return new THREE.Vector3(bankX(k)+((ix%cols)-(cols-1)/2)*.55,
    .25-Math.floor(ix/cols)*.45,Math.sin(ix*2.4)*.19);
}
function channelPosition(k,id){
  if(isRGB())return featurePosition(id);
  // Schematic channel placement, deterministic by channel ID. Not die coordinates.
  const a=(id*2.399963229728653),r=.48+Math.sqrt((id%109)/109)*1.05;
  return new THREE.Vector3(bankX(k)+Math.cos(a)*r,(isRGB()?2.9:2.55)+Math.sin(a)*r*(isRGB()?.3:.66),Math.sin(id*.71)*.7);
}
function makeLine(from,to,color,opacity=.4){
  const mid=from.clone().lerp(to,.5);mid.z+=.25;const curve=new THREE.QuadraticBezierCurve3(from,mid,to);
  const g=new THREE.BufferGeometry().setFromPoints(curve.getPoints(12));
  const line=new THREE.Line(g,new THREE.LineBasicMaterial({color,transparent:true,opacity,depthWrite:false}));dynamic.add(line);
}
function rebuildScene(){
  const ss=stateAt(side==='before'),e=event();if(!ss||!e)return;
  clearScene();outputTrace=null;const showHelpers=$('helpers').checked,showConnections=!nodeView||$('nodeConnections').checked,allLinks=nodeView?!isRGB():$('links').value==='all';
  floor.visible=!isRGB()&&!nodeView&&backgroundOpacity===1;setting.visible=isRGB()&&!nodeView&&backgroundOpacity===1;
  for(let k=0;k<data.banks.length;k++){
    const dim=data.banks[k],ids=showHelpers?Array.from({length:dim.units},(_,i)=>i):dim.plastic;
    const dimmed=!allBanks&&k!==bank,baseColor=new THREE.Color(colors[k]);
    label(new THREE.Vector3(bankX(k),isRGB()?3.25:4,0),bankNames[k],dimmed?'#52627a':colors[k],true);
    label(new THREE.Vector3(bankX(k),isRGB()?2.98:3.65,0),`NPU ${dim.mapping.join('.')}`,dimmed?'#52627a':'#91a5b8');

    const channelIds=new Set();
    for(const j of ids)if(allLinks||(k===bank&&j===record))ss[k][j].connections.forEach(c=>channelIds.add(c));
    if(k===bank){selectedChange()?.removed.forEach(c=>channelIds.add(c));selectedChange()?.added.forEach(c=>channelIds.add(c));}
    const active=new Set(e.derived_features[k]);
    if(isRGB()&&!nodeView){dynamic.add(inputVolume(active,dim.inputs,hoverChannel));label(new THREE.Vector3(0,1.55,.1),`INPUTS ${active.size}/${dim.inputs} · DERIVED`,'#75adc4');}
    if(isRGB()&&k===bank&&hoverChannel!==null)channelIds.add(hoverChannel);
    for(const channel of showConnections?channelIds:[]){
      const p=channelPosition(k,channel),isHover=hoverChannel===channel&&bank===k;
      const node=new THREE.Mesh(inputGeometry,new THREE.MeshBasicMaterial({color:isHover?'#ffffff':active.has(channel)?colors[k]:'#30435a',transparent:true,opacity:dimmed?.15:active.has(channel)?.9:.45}));
      node.userData={kind:'channel',bank:k,channel,shared:true};node.position.copy(p);dynamic.add(node);targets.push(node);
      if(k===bank&&isHover)label(p.clone().add(new THREE.Vector3(0,.17,0)),`input ${channel}`,'#e7edf5');
    }
    for(const j of ids){
      const pos=neuronPosition(k,j,ids),sel=k===bank&&j===record;
      const gain=isRGB()?e.response_gain[k][j]:0,magnitude=isRGB()?Math.sqrt(Math.min(Math.abs(gain)/32,1)):0;
      const emitted=isRepeated()?e.native_responses[k]?.[j]===1:isRGB()?gain>0:e.derived_potentials[k][j]>dim.thresholds[j];
      const linked=hoverChannel!==null&&bank===k&&ss[k][j].connections.includes(hoverChannel);
      const change=e.changes.find(c=>c.bank===k&&c.record===j);
      const responseColor=isRGB()&&gain<0?new THREE.Color('#ef858c'):baseColor;
      const mat=new THREE.MeshPhysicalMaterial({color:responseColor.clone().multiplyScalar(isRepeated()?(emitted?1.25:.3):dimmed?.25:sel?1:isRGB()?.3+.7*magnitude:emitted?.95:.48),
        roughness:.22,metalness:.28,clearcoat:.8,clearcoatRoughness:.15,emissive:isRepeated()?(emitted?colors[k]:'#000000'):sel?'#28619c':linked?'#7be1c0':isRGB()?responseColor:emitted?colors[k]:'#000000',emissiveIntensity:isRepeated()?(emitted?.8:0):sel?.65:linked?.65:isRGB()?magnitude*.7:emitted?.3:0});
      const mesh=new THREE.Mesh(sphere,mat);mesh.position.copy(pos);mesh.scale.setScalar(sel?1.5:isRGB()?.6+.65*magnitude:1);
      mesh.userData={kind:'neuron',bank:k,record:j,shared:true};dynamic.add(mesh);targets.push(mesh);
      if(isRGB()){const halo=responseHalo(pos,change?'#7be1c0':responseColor,magnitude,!!change);if(halo)dynamic.add(halo);}
      if(sel||change){const ring=new THREE.Mesh(new THREE.TorusGeometry(sel?.2:.15,.008,5,40),new THREE.MeshBasicMaterial({color:sel?'#e7edf5':'#7be1c0',transparent:true,opacity:.8}));ring.position.copy(pos);dynamic.add(ring);}
      if(!dimmed&&(ids.length<=16||sel))label(pos.clone().add(new THREE.Vector3(0,-.22,0)),String(j).padStart(2,'0'),sel?'#ffffff':'#8ea1b8');
      if(showConnections&&(allLinks||sel)){
        for(const c of ss[k][j].connections){
          const added=change?.added.includes(c),removed=change?.removed.includes(c);
          makeLine(channelPosition(k,c),pos,added?'#7be1c0':removed?'#ef858c':colors[k],dimmed?.025:sel?.65:.085);
        }
        if(sel)for(const c of change?.removed||[])if(!ss[k][j].connections.includes(c))makeLine(channelPosition(k,c),pos,'#ef858c',.65);
      }
    }
  }
  if(isRGB()&&(!nodeView||$('nodeConnections').checked)&&($('helpers').checked||data.banks[0].plastic.includes(record))){
    const response=neuronResponse(data,e,record),ids=$('helpers').checked?Array.from({length:data.banks[0].units},(_,i)=>i):data.banks[0].plastic;
    outputTrace=responseTrace(neuronPosition(0,record,ids),response);dynamic.add(outputTrace.group);
    const p=outputTrace.endpoint.clone().add(new THREE.Vector3(0,0,.08));
    label(p.clone().add(new THREE.Vector3(0,.48,0)),'SELECTED NEURON OUTPUT','#c5a9ff');
    label(p.clone().add(new THREE.Vector3(0,.13,0)),`VIS ${String(record).padStart(3,'0')} → ${response.value}/${response.max}`,'#f1e8ff',true);
    label(p.clone().add(new THREE.Vector3(0,-.13,0)),response.usedByClassReadout?'Captured · pre-update':'Helper · outside readout','#bca7dc');
  }
  if(isRGB())for(let g=0;g<($('helpers').checked?11:10);g++){
    const a=g*Math.PI*2/11+Math.PI/2;label(new THREE.Vector3(Math.cos(a)*3.35,.9+Math.sin(a)*1.65,Math.sin(g*1.7)*.57),g===10?'Protected queries':`Object ${g+1}`,g===10?'#91a5b8':colors[0]);
  }
}
function resize(){const r=$('sceneWrap').getBoundingClientRect();renderer.setSize(r.width,r.height,false);camera.aspect=r.width/r.height;camera.updateProjectionMatrix();drawMatrix();drawTimeline();}
function fitView(){const r=$('sceneWrap').getBoundingClientRect(),aspect=r.width/r.height;
  if(!dynamic.children.length||!r.width||!r.height)return;
  camera.aspect=aspect;camera.updateProjectionMatrix();
  const bounds=new THREE.Box3().setFromObject(dynamic);
  if(!nodeView||$('nodeLabels').checked)for(const item of labels)bounds.expandByPoint(item.position);
  const center=bounds.getCenter(new THREE.Vector3()),direction=new THREE.Vector3(1,3,14).normalize(),corners=[];
  for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z])corners.push(new THREE.Vector3(x,y,z));
  controls.target.copy(center);let distance=7;
  for(let n=0;n<50;n++){
    camera.position.copy(direction).multiplyScalar(distance).add(center);camera.lookAt(center);camera.updateMatrixWorld();
    if(corners.every(c=>{const p=c.clone().project(camera);return Math.abs(p.x)<.84&&Math.abs(p.y)<.82&&p.z<1;}))break;
    distance*=1.1;
  }
  controls.maxDistance=Math.max(32,distance*1.5);controls.update();}
new ResizeObserver(resize).observe($('sceneWrap'));
let pointerStart=null;
$('scene').addEventListener('pointerdown',e=>pointerStart=[e.clientX,e.clientY]);
$('scene').addEventListener('pointerup',e=>{
  if(!pointerStart||Math.hypot(e.clientX-pointerStart[0],e.clientY-pointerStart[1])>5)return;
  const r=$('scene').getBoundingClientRect();mouse.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);raycaster.setFromCamera(mouse,camera);
  const hit=raycaster.intersectObjects(targets)[0];if(!hit){$('nodeSelection').hidden=true;return;}
  const u=hit.object.userData;bank=u.bank;
  if(u.kind==='neuron'){record=u.record;$('followEdits').checked=false;}else{hoverChannel=u.channel;$('channel').value=u.channel;}
  render();if(nodeView&&u.kind==='neuron'){$('nodeSelection').hidden=false;nodeSelection();}
});
function nodeSelection(){
  if(!event()||!stateAt())return;
  $('nodeTitle').textContent=`${bankNames[bank]} · record ${String(record).padStart(2,'0')}`;
  const change=selectedChange(),r=stateAt(side==='before')[bank][record];
  $('nodeSummary').textContent=`${r.connections.length} stored connections · +${change?.added.length||0} / −${change?.removed.length||0} this event`;
  $('nodeSelectionEvidence').textContent=`Event ${index+1} · NPU (${data.banks[bank].mapping.join(', ')}) · Captured RAM${isRGB()?`; neuron response ${recordScore(bank,record)}/${bankBudget()}`:''}; ${isRepeated()?'core: returned response; ring: selection/edit':'glow reconstructed'}.`;
}
function setBackground(value){
  const n=Number(value);if(!Number.isFinite(n))return;
  backgroundOpacity=Math.max(0,Math.min(100,n))/100;
  renderer.setClearAlpha(backgroundOpacity);floor.visible=!isRGB()&&!nodeView&&backgroundOpacity===1;setting.visible=isRGB()&&!nodeView&&backgroundOpacity===1;
  const percent=Math.round(backgroundOpacity*100);
  $('backgroundOpacity').value=$('nodeOpacity').value=percent;
  $('opacityValue').textContent=$('nodeOpacityValue').textContent=`${percent}%`;
  $('backdrop').value=[0,40,100].includes(percent)?String(percent):'custom';
  $('sceneWrap').classList.toggle('alphaPreview',backgroundOpacity<1);
  $('scene').dataset.backgroundOpacity=String(backgroundOpacity);
  $('capture').textContent=backgroundOpacity<1?'Save alpha PNG':'Save view';
}
function setNodeView(enabled){
  nodeView=enabled;document.body.classList.toggle('node-view',enabled);$('nodeOverlay').hidden=!enabled;
  $('nodeTools').hidden=true;$('nodeSelection').hidden=true;$('nodeMenu').setAttribute('aria-expanded','false');
  floor.visible=!enabled&&backgroundOpacity===1;
  if(data)rebuildScene();resize();fitView();$('scene').focus({preventScroll:true});
}
$('backdrop').onchange=()=>{if($('backdrop').value!=='custom')setBackground($('backdrop').value);};
$('backgroundOpacity').oninput=e=>setBackground(e.target.value);
$('nodeOpacity').oninput=e=>setBackground(e.target.value);
$('nodeView').onclick=()=>setNodeView(true);
$('exitNodeView').onclick=()=>setNodeView(false);
$('nodeMenu').onclick=()=>{$('nodeTools').hidden=!$('nodeTools').hidden;$('nodeMenu').setAttribute('aria-expanded',String(!$('nodeTools').hidden));};
$('nodeConnections').onchange=()=>{rebuildScene();fitView();};
$('nodeLabels').onchange=()=>document.body.classList.toggle('node-labels',$('nodeLabels').checked);
$('nodeSolid').onclick=()=>setBackground(100);$('nodeGlass').onclick=()=>setBackground(40);$('nodeClear').onclick=()=>setBackground(0);
$('closeNodeSelection').onclick=()=>{$('nodeSelection').hidden=true;};
$('nodeInspect').onclick=()=>{setNodeView(false);document.querySelector('.inspector').scrollIntoView({behavior:'smooth',block:'start'});};
$('nodeReset').onclick=fitView;
function bankPicker(){
  $('bankPicker').replaceChildren();
  for(const [k,b] of data.banks.entries()){
    const wrap=document.createElement('div');wrap.className='bankItem';const title=document.createElement('button');title.className='bankName';
    title.style.color=colors[k];title.textContent=b.name;const meta=document.createElement('span');meta.textContent=b.plastic.length+' plastic';title.append(meta);
    title.onclick=()=>{bank=k;record=b.plastic[0];allBanks=false;$('followEdits').checked=false;render();};wrap.append(title);
    const buttons=document.createElement('div');buttons.className='neuronButtons';
    for(const j of b.plastic){const button=document.createElement('button');button.textContent=String(j).padStart(2,'0');button.title=`${b.name} record ${j}`;button.setAttribute('aria-label',button.title);button.classList.toggle('selected',k===bank&&j===record);button.onclick=()=>{bank=k;record=j;$('followEdits').checked=false;render();};buttons.append(button);}
    wrap.append(buttons);$('bankPicker').append(wrap);
  }
}
function inspect(){
  const ss=stateAt(side==='before'),e=event();if(!ss||!e)return;
  const b=data.banks[bank],r=ss[bank][record],delta=selectedChange(),plastic=b.plastic.includes(record);$('channel').max=bankInputs()-1;$('channel').placeholder=`0–${bankInputs()-1}`;
  $('recordTitle').textContent=`${isRGB()?'VIS':['POL','MEM0','MEM1','TRACE'][bank]} : ${String(record).padStart(2,'0')}`;
  $('recordLocation').textContent=`${b.name} · NPU (${b.mapping.join(', ')}) · ${side==='before'?'Before':'After'} event ${index+1}`;
  $('before').classList.toggle('active',side==='before');$('after').classList.toggle('active',side==='after');
  const group=Math.floor(record/(b.units/b.classes));
  const stats=[['Record role',plastic?'Plastic':'Fixed helper'],['Learning class',isRGB()?(group===10?'Protected queries':`Object ${group+1} (class ${group})`):`${group}`],['Input connections',`${r.connections.length} / ${b.inputs||512}`],['Learning threshold',r.words[3]>>>16],['Swap budget',r.words[3]&65535],['Changes this event',`${delta?.added.length||0} in / ${delta?.removed.length||0} out`]];
  $('recordStats').replaceChildren(...stats.map(([a,b])=>{const el=document.createElement('div');el.className='stat';const small=document.createElement('small'),val=document.createElement('b');small.textContent=a;val.textContent=b;el.append(small,val);return el;}));
  const channels=[...new Set([...r.connections,...(delta?.removed||[])])].sort((a,b)=>a-b);
  $('connectionList').replaceChildren(...channels.map(c=>{const button=document.createElement('button');button.textContent=c;button.title=`Input channel ${c}`;button.classList.toggle('new',delta?.added.includes(c)||false);button.classList.toggle('gone',!r.connections.includes(c));button.classList.toggle('highlight',c===hoverChannel);button.onclick=()=>{hoverChannel=c;$('channel').value=c;render();};return button;}));
  $('connectionCount').textContent=r.connections.length;
  $('connectionDiff').textContent=delta?`Added: ${delta.added.join(', ')||'none'}. Removed: ${delta.removed.join(', ')||'none'}.`:'No captured connection or record-word change for this neuron at this event.';
  $('recordWords').textContent=r.words.map((v,j)=>`word ${j}  0x${v.toString(16).toUpperCase().padStart(8,'0')}`).join('\n');
  const overlap=recordScore(bank,record),threshold=b.thresholds?.[record];
  $('overlap').textContent=`${isRGB()?'Captured match score':'Pre-edit overlap'}: ${overlap} / ${bankBudget()}`;
  $('overlapBar').style.width=`${overlap/bankBudget()*100}%`;
  $('unitOutput').textContent=isRGB()?`Change versus initial mask: ${e.response_gain[bank][record]>=0?'+':''}${e.response_gain[bank][record]} (derived). Captured neuron output: ${overlap}/${bankBudget()} (pre-update). No electrical spike measurement.`:`Derived unit output: ${overlap>threshold?1:0}. Inference threshold: > ${threshold}.`;
}
function drawMatrix(){
  if(!data||!event())return;const canvas=$('matrix'),rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio,2);
  canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const b=data.banks[bank],ss=stateAt(side==='before')[bank],ids=$('helpers').checked?Array.from({length:b.units},(_,i)=>i):b.plastic;
  const left=40,w=rect.width-left-12,row=(rect.height-25)/ids.length,cw=w/bankInputs();
  ctx.fillStyle='#0b1420';ctx.fillRect(0,0,rect.width,rect.height);ctx.font='9px monospace';
  for(let p=0;p<ids.length;p++){
    const j=ids[p],yy=20+p*row,delta=event().changes.find(c=>c.bank===bank&&c.record===j);
    if(j===record){ctx.fillStyle='#1d3148';ctx.fillRect(0,yy-1,rect.width,row);}
    ctx.fillStyle='#91a5b8';ctx.fillText(String(j).padStart(2,'0'),12,yy+Math.min(9,row-1));
    for(const c of ss[j].connections){ctx.fillStyle=delta?.added.includes(c)?'#7be1c0':colors[bank];ctx.fillRect(left+c*cw,yy,Math.max(cw,1),Math.max(row-2,1));}
    for(const c of delta?.removed||[]){ctx.fillStyle='#ef858c';ctx.fillRect(left+c*cw,yy,Math.max(cw,1),Math.max(row-2,1));}
  }
  for(let c=0;c<=bankInputs();c+=bankInputs()/8){ctx.fillStyle='#91a5b8';ctx.fillText(c===bankInputs()?String(bankInputs()-1):c,left+Math.min(c,bankInputs()-1)*cw,10);}
  if(hoverChannel!==null){ctx.strokeStyle='#ffffff';ctx.strokeRect(left+hoverChannel*cw,17,Math.max(cw,1),rect.height-20);}
  $('matrixTitle').textContent=`${b.name} / ${ids.length} ${$('helpers').checked?'configured':'plastic'} records × ${bankInputs()} inputs`;
}
$('matrix').onclick=e=>{const r=$('matrix').getBoundingClientRect(),ids=$('helpers').checked?Array.from({length:data.banks[bank].units},(_,i)=>i):data.banks[bank].plastic;
  const row=Math.floor((e.clientY-r.top-20)/((r.height-25)/ids.length));if(row>=0&&row<ids.length)record=ids[row];
  const c=Math.floor((e.clientX-r.left-40)/(r.width-52)*bankInputs());if(c>=0&&c<bankInputs()){hoverChannel=c;$('channel').value=c;}render();};
function drawTimeline(){
  if(!data||!events().length)return;const c=$('editTimeline'),r=c.getBoundingClientRect();c.width=Math.round(r.width*devicePixelRatio);c.height=Math.round(r.height*devicePixelRatio);const ctx=c.getContext('2d');
  const es=events();for(let i=0;i<es.length;i++)for(let k=0;k<data.banks.length;k++)if(es[i].counts[k].added){ctx.fillStyle=colors[k];ctx.fillRect(i/es.length*c.width,k*c.height/data.banks.length,Math.max(c.width/es.length,1),c.height/data.banks.length);}
}
function comparison(){
  if(isRepeated()){renderRepeatedComparison($('comparison'),data);return;}
  if(isRGB()){
    const table=document.createElement('table');table.innerHTML='<thead><tr><th>Arm</th><th>Initial</th><th>New views</th><th>Return</th></tr></thead>';const body=document.createElement('tbody');
    const entries=Object.values(data.arms).map(a=>[a.label,a.assessment]);
    for(const [key,label] of [['nearest_hamming_adaptive','CPU nearest'],['cosine_class_mean_adaptive','CPU class mean']])if(data.cpu_controls?.[key])entries.push([label,data.cpu_controls[key]]);
    for(const [name,metrics] of entries){const tr=document.createElement('tr');const td=document.createElement('td');td.textContent=name;tr.append(td);for(const key of ['old','after','retention']){const cell=document.createElement('td'),v=metrics[key];cell.textContent=`${v.correct}/${v.total}`;tr.append(cell);}body.append(tr);}table.append(body);$('comparison').replaceChildren(table);return;
  }
  const phases=['protected_exam','updated_exam'].map(prefix=>Object.keys(data.arms.native.assessment).find(k=>k.startsWith(prefix)));const table=document.createElement('table');table.innerHTML='<thead><tr><th>Arm</th><th>Protected</th><th>Updated</th></tr></thead>';const body=document.createElement('tbody');
  for(const arm of Object.values(data.arms)){const tr=document.createElement('tr');const td=document.createElement('td');td.textContent=arm.label;tr.append(td);
    for(const ph of phases){const v=arm.assessment[ph],cell=document.createElement('td');cell.textContent=`${v.memory_correct}/${v.queries}`;tr.append(cell);}body.append(tr);}
  const tr=document.createElement('tr');tr.innerHTML='<td>Adaptive CPU</td>';for(const ph of phases){const v=data.arms.native.assessment[ph],td=document.createElement('td');td.textContent=`${v.cpu.adaptive_dictionary}/${v.queries}`;tr.append(td);}body.append(tr);table.append(body);$('comparison').replaceChildren(table);
}
function render(){
  const e=event();if(!e)return;index=Math.min(index,events().length-1);
  if(isRepeated()){
    // Bind the existing image/output panels after the shared RAM inspector below.
  }else if(isRGB()){
    const phases={prefix:'Teach initial views',old:'Recall initial views',before:'Query changed views',adapt:'Teach changed views',after:'Recall changed views',retention:'Return to initial views'};
    const image=data.images[e.image_id];$('visualImage').src=image.data_url;$('visualImage').alt=`Recorded object ${e.evaluation_target+1}, session ${image.session}, frame ${image.frame}`;
    $('visualImageMeta').textContent=`CORe50 · session ${image.session} · frame ${image.frame} · 128×128 RGB`;
    $('visualMode').textContent=e.teacher_enabled?'TEACHING ENABLED':'READ ONLY · NO TEACHING';$('visualMode').classList.toggle('teaching',e.teacher_enabled);
    $('visualTarget').textContent=`${e.teacher_enabled?'Teaching identity':'Evaluation target'}: Object ${e.evaluation_target+1}`;
    $('visualExplanation').textContent=e.teacher_enabled?'The image encoder stays fixed. Native memory may change its stored connections.':'Target identity is for scoring only. The image is queried without a teaching label; captured memory stays unchanged.';
    $('phaseTitle').textContent=phases[e.phase]||pretty(e.phase);$('eventDescription').textContent=`${e.teacher_enabled?'Teach':'Query'} · object ${e.evaluation_target+1}`;
    $('eventDetail').textContent=`${currentArm().label}. 20 fixed CNPs + 1 memory FNP3, plus HRC. The two arms are separate executions.`;
    $('decision').textContent=e.teacher_enabled?'TEACH':'READ ONLY';$('decision').style.color=e.teacher_enabled?'#e1bf7b':'#6eb8ff';$('decisionDetail').textContent='No manager/admission output is claimed for this profile.';
    renderMemoryPanel($('visualMemoryPanel'),{data,arm:currentArm(),index,side,state:stateAt(side==='before'),record,onSelect:j=>{bank=0;record=j;$('followEdits').checked=false;render();}});
    $('memoryAnswer').textContent=`Object ${e.prediction+1}`;$('memoryAnswer').style.color=e.prediction===e.evaluation_target?'#7be1c0':'#ef858c';$('cpuAnswer').textContent=`Class argmax from captured native scores. ${e.prediction===e.evaluation_target?'Correct':'Incorrect'} for this image.`;
  }else{
  $('phaseTitle').textContent=pretty(e.phase);$('eventDescription').textContent=`${pretty(e.role)} · fact ${e.key}`;
  $('eventDetail').textContent=e.role==='store'?`Native trace storage: source ${e.policy_key}, candidate ${e.candidate}; no outcome supplied.`:e.role==='lookup'?`Read-only native trace lookup. Returned joint-class flags [${(e.trace_flags||[]).join(', ')}].`:e.delayed?`Later outcome ${e.outcome}; candidate and policy fields are not rewritten. The response comes from native trace memory.`:e.role.startsWith('feedback')?`Candidate ${e.candidate}, observed outcome ${e.outcome}. External supervision trains the manager.`:e.role==='transfer'?`Candidate ${e.candidate} from source/context ${e.policy_key}. Native routing controls student input.`:'Read-only evaluation; all bank learning clocks disabled.';
  let q=e;if(e.role==='transfer')q=events()[index-1];const applicable=['query','transfer'].includes(e.role);
  $('decision').textContent=applicable?({0:'REJECT',1:'ADMIT'}[decode(q.policy_flags)]||decode(q.policy_flags)):'Not queried';
  $('decision').style.color=applicable&&decode(q.policy_flags)==='0'?'#ef858c':'#7be1c0';
  $('decisionDetail').textContent=applicable?`Flags [${q.policy_flags.join(', ')}] · returned event ${q.index+1}`:'No admission decision attributed to this phase.';
  $('memoryAnswer').textContent=e.role==='exam'?decode(e.memory_flags):'—';
  $('cpuAnswer').textContent=e.role==='exam'?`Adaptive CPU: ${e.cpu_memory[0]<0?'No match':e.cpu_memory[0]}. ${mode==='replay'?`Reference answer: ${e.actual}.`:''}`:'Memory recall is assessed on query-only exam events.';
  }
  $('editCounts').replaceChildren(...e.counts.map((c,k)=>{const span=document.createElement('span');span.style.color=colors[k];span.textContent=`${bankNames[k]}  +${c.added} / −${c.removed}`;return span;}));
  $('eventCounter').textContent=`Event ${index+1} / ${events().length}`;$('phaseCounter').textContent=pretty(e.role);
  $('timeline').max=events().length-1;$('timeline').value=index;evidenceLabel();
  bankPicker();inspect();if(isRepeated())renderRepeatedEvent({data,arm:currentArm(),e,state:stateAt(side==='before'),before:stateAt(true),side,bank,record,onSelect:j=>{bank=1;record=j;$('followEdits').checked=false;render();}});nodeSelection();drawMatrix();drawTimeline();rebuildScene();
}
function selectIndex(n){index=Math.max(0,Math.min(events().length-1,n));if($('followEdits').checked){if(event().changes.length){const c=event().changes[0];bank=c.bank;record=c.record;}else if(isRGB()){bank=0;const scores=event().observed_potentials[0].slice(0,200);record=scores.indexOf(Math.max(...scores));}}render();}
$('arm').onchange=()=>{if(mode==='live')return;arm=$('arm').value;if(isRGB())colors[0]=arm==='adaptive'?'#e1bf7b':'#6eb8ff';caches.delete(arm);stopPlayback();render();};
$('timeline').oninput=()=>{stopPlayback();selectIndex(Number($('timeline').value));};
$('previous').onclick=()=>{stopPlayback();selectIndex(index-1);};$('next').onclick=()=>{stopPlayback();selectIndex(index+1);};
$('play').onclick=()=>{playing=!playing;playbackLabels();};
$('nodePlay').onclick=()=>$('play').click();
$('nextEdit').onclick=()=>{stopPlayback();const es=events();let n=es.findIndex((e,i)=>i>index&&e.changes.some(c=>c.added.length||c.removed.length));if(n<0)n=es.findIndex(e=>e.changes.some(c=>c.added.length||c.removed.length));if(n>=0){const c=es[n].changes.find(c=>c.added.length||c.removed.length);bank=c.bank;record=c.record;selectIndex(n);}};
$('before').onclick=()=>{side='before';render();};$('after').onclick=()=>{side='after';render();};
$('showAll').onclick=()=>{allBanks=true;render();};$('helpers').onchange=render;$('links').onchange=render;
$('channel').oninput=()=>{const v=$('channel').value;hoverChannel=v===''?null:Math.max(0,Math.min(bankInputs()-1,Number(v)));render();};
$('resetView').onclick=fitView;
function captureView(){
  renderer.render(scene,camera);const caption=!nodeView&&backgroundOpacity===1;
  const c=document.createElement('canvas');c.width=renderer.domElement.width;c.height=renderer.domElement.height+(caption?100:0);
  const ctx=c.getContext('2d');
  if(caption){ctx.fillStyle='#080e17';ctx.fillRect(0,0,c.width,c.height);}
  ctx.drawImage(renderer.domElement,0,0);
  if(caption){ctx.fillStyle='#e7edf5';ctx.font='22px sans-serif';ctx.fillText(`AKD1500 / ${currentArm().label||'Live native trial'} / event ${index+1}`,24,c.height-62);
    ctx.fillStyle='#91a5b8';ctx.font='16px sans-serif';ctx.fillText(isRGB()?'Captured connections and scores; derived response gain. Logical geometry, no electrical spike timing.':'Captured connections; reconstructed unit output. Logical geometry; no electrical spike timing.',24,c.height-30);}
  c.toBlob(blob=>{if(!blob){error('PNG export unavailable');return;}const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`akd-captured-connections-reconstructed-output-event-${index+1}-alpha-${Math.round(backgroundOpacity*100)}.png`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
}
$('capture').onclick=captureView;$('nodeSave').onclick=captureView;
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON','TEXTAREA'].includes(document.activeElement.tagName))return;
  if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.key==='ArrowRight')$('next').click();if(e.key==='ArrowLeft')$('previous').click();});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&nodeView){e.preventDefault();setNodeView(false);}});
function switchMode(next){if((isRGB()||isRepeated())&&next==='live')return;stopPlayback();mode=next;$('replayMode').classList.toggle('active',mode==='replay');$('liveMode').classList.toggle('active',mode==='live');$('liveControls').hidden=mode!=='live';$('arm').disabled=mode==='live'&&live?.status==='ready';
  if(mode==='replay'){arm=$('arm').value;index=Math.min(index,events().length-1);$('sourceBadge').textContent=`SOURCE VERIFIED · ${Object.keys(data.arms).length} HARDWARE ARMS`;render();}
  else{$('sourceBadge').textContent='SEPARATE LIVE TRIAL';if(live?.events.length)index=live.events.length-1;render();$('liveStatus').textContent=live?.status||'No hardware session. Start explicitly teaches the known prefix.';}}
$('replayMode').onclick=()=>switchMode('replay');$('liveMode').onclick=()=>switchMode('live');
function liveButtons(){const ready=live?.status==='ready'&&!busy;$('startLive').disabled=busy||['ready','busy','starting','bootstrapping'].includes(live?.status)||!status?.native_enabled;
  for(const id of ['queryLive','proposeLive','feedbackLive'])$(id).disabled=!ready;
  $('stopLive').disabled=busy||!live?.initial;$('arm').disabled=mode==='live'&&['ready','busy','starting','bootstrapping'].includes(live?.status);
}
async function command(a){const previous=live?.events?.length||0;busy=true;liveButtons();$('error').hidden=true;
  try{const r=await fetch('/api/live',{method:'POST',headers:{'Content-Type':'application/json','X-Explorer-Token':status.csrf},body:JSON.stringify(a)});const v=await r.json();if(!r.ok)throw Error(v.error);live=v;caches.delete('live');
    if(mode==='live'&&live.events?.length){index=live.events.length-1;
      if(clip){const changed=live.events.find((e,i)=>i>=previous&&e.changes.length);if(changed){index=changed.index;bank=changed.changes[0].bank;record=changed.changes[0].record;}}
      render();}}
  catch(e){error(e);}finally{busy=false;liveButtons();}}
$('startLive').onclick=()=>command({action:'start',arm:$('arm').value,mode:'native'});
$('stopLive').onclick=()=>command({action:'stop'});
$('queryLive').onclick=()=>command({action:'query',key:Number($('fact').value)});
$('proposeLive').onclick=()=>command({action:'propose',key:Number($('fact').value),policy:Number($('policy').value),candidate:Number($('candidate').value)});
$('feedbackLive').onclick=()=>command({action:'feedback',policy:Number($('policy').value),candidate:Number($('candidate').value),outcome:Number($('outcome').value)});
async function poll(){
  if(isRGB()||isRepeated())return;
  try{const r=await fetch('/api/live');const v=await r.json();const changed=v.events?.length!==live?.events?.length;live=v;
    $('liveStatus').textContent=`${v.status} · ${v.events?.length||0} captured events${v.mode?' · '+v.mode:''}${v.error?' · '+v.error:''}`;evidenceLabel();
    if(mode==='live'&&changed&&v.events.length){caches.delete('live');index=v.events.length-1;render();}liveButtons();
  }catch(e){$('liveStatus').textContent='Connection unavailable; hardware state not assumed.';}
}
function animate(t){requestAnimationFrame(animate);
  controls.update();renderer.render(scene,camera);
  drawRecording(t);
  const r=$('sceneWrap').getBoundingClientRect();for(const {el,position} of labels){const p=position.clone().project(camera);el.hidden=p.z>1||p.z< -1;el.style.left=`${(p.x+1)/2*r.width}px`;el.style.top=`${(-p.y+1)/2*r.height}px`;}
  if(playing&&t-lastTick>1000/Number($('speed').value)){lastTick=t;if(index<events().length-1)selectIndex(index+1);else stopPlayback();}
}
let clip=null,preparingClip=false,completedClipURLs=[];
function drawRGBRecording(){
  const {ctx,rgbRenderer,rgbCamera,center,direction,distance,imageCache,start}=clip,e=event(),elapsed=(performance.now()-start)/1000;
  ctx.fillStyle='#080e17';ctx.fillRect(0,0,1920,1080);
  ctx.fillStyle='#e7edf5';ctx.font='500 34px sans-serif';ctx.fillText('AKD1500 / Looking inside visual memory',54,62);
  ctx.fillStyle='#7be1c0';ctx.font='18px monospace';ctx.fillText(`RECORDED CHIP REPLAY  ·  ${currentArm().label.toUpperCase()}  ·  EVENT ${index+1} / 520`,54,99);
  ctx.strokeStyle='#26374b';ctx.beginPath();ctx.moveTo(54,126);ctx.lineTo(1866,126);ctx.stroke();
  ctx.fillStyle='#91a5b8';ctx.font='16px sans-serif';ctx.fillText('ACTUAL RECORDED RGB INPUT',54,166);
  const image=imageCache.get(e.image_id);if(image?.complete)ctx.drawImage(image,54,190,324,324);
  ctx.fillStyle='#e7edf5';ctx.font='24px sans-serif';ctx.fillText(e.teacher_enabled?'Teach changed views':'Query without teaching',54,560);
  ctx.fillStyle=e.teacher_enabled?'#e1bf7b':'#6eb8ff';ctx.font='21px sans-serif';ctx.fillText(`${e.teacher_enabled?'Teaching identity':'Evaluation target'}: Object ${e.evaluation_target+1}`,54,598);
  ctx.fillStyle='#91a5b8';ctx.font='16px sans-serif';ctx.fillText(`CORe50 · session ${data.images[e.image_id].session} · frame ${data.images[e.image_id].frame}`,54,635);
  ctx.fillText('128 × 128 RGB · fixed image encoder',54,662);
  ctx.fillStyle='#e7edf5';ctx.font='21px sans-serif';ctx.fillText('Changed-view recall',54,740);
  ctx.fillStyle='#e1bf7b';ctx.font='28px monospace';ctx.fillText('44 / 60 learned',54,785);
  ctx.fillStyle='#6eb8ff';ctx.font='24px monospace';ctx.fillText('26 / 60 frozen',54,825);
  ctx.fillStyle='#91a5b8';ctx.font='16px sans-serif';ctx.fillText('Full recording comparison; correlated views.',54,861);ctx.fillText('Initial recall 80/120: below the 70% gate.',54,887);
  rgbCamera.position.copy(direction).multiplyScalar(distance).add(center);rgbCamera.lookAt(center);rgbRenderer.render(scene,rgbCamera);ctx.drawImage(rgbRenderer.domElement,410,160,1030,730);
  for(const {el,position} of labels){const p=position.clone().project(rgbCamera);if(p.z>1||p.z< -1)continue;ctx.fillStyle=el.style.color||'#91a5b8';ctx.font=(el.querySelector('strong')?'600 22px':'15px')+' sans-serif';ctx.textAlign='center';ctx.fillText(el.textContent,410+(p.x+1)/2*1030,160+(-p.y+1)/2*730);}
  ctx.textAlign='left';const x=1480,rec=stateAt()[bank][record],delta=selectedChange();
  const summary=memorySummary(data,currentArm(),index,side),winner=summary.winner;
  ctx.fillStyle='#91a5b8';ctx.font='16px sans-serif';ctx.fillText('READOUT · HOST FROM CHIP SCORES',x,166);
  const reference=winner.reference?imageCache.get(winner.reference):null;if(reference?.complete)ctx.drawImage(reference,x,188,74,74);
  ctx.fillStyle=e.prediction===e.evaluation_target?'#7be1c0':'#ef858c';ctx.font='30px sans-serif';ctx.fillText(winner.label,x+92,216);
  ctx.fillStyle='#c8d5e4';ctx.font='17px monospace';ctx.fillText(`${winner.score}/64 native match`,x+92,249);
  ctx.fillStyle='#91a5b8';ctx.font='13px sans-serif';ctx.fillText('Thumbnail: host teaching reference',x,286);
  summary.groups.forEach((g,j)=>{const cx=x+(j%2)*196,cy=317+Math.floor(j/2)*22;ctx.font='12px sans-serif';ctx.fillStyle='#91a5b8';ctx.fillText(g.label,cx,cy);ctx.fillStyle='#24374b';ctx.fillRect(cx+65,cy-8,86,4);ctx.fillStyle=g.id===winner.id?'#7be1c0':'#607e99';ctx.fillRect(cx+65,cy-8,86*g.score/64,4);ctx.font='12px monospace';ctx.fillText(String(g.score),cx+160,cy);});
  ctx.fillStyle='#91a5b8';ctx.font='13px sans-serif';ctx.fillText('Pre-update matches; not probabilities.',x,435);
  ctx.font='16px sans-serif';ctx.fillText('OBJECT MEMORY · CAPTURED STATE',x,477);
  summary.groups.forEach((g,j)=>{const cx=x+j%5*77,cy=498+Math.floor(j/5)*91,im=g.reference?imageCache.get(g.reference):null;ctx.strokeStyle=g.id===e.prediction?'#7be1c0':'#293d50';ctx.strokeRect(cx-3,cy-3,68,81);if(im?.complete)ctx.drawImage(im,cx,cy,62,52);ctx.fillStyle='#c8d5e4';ctx.font='11px sans-serif';ctx.fillText(g.label,cx+3,cy+65);ctx.fillStyle='#91a5b8';ctx.font='9px monospace';ctx.fillText(`${g.updated}/20 updated`,cx+1,cy+76);});
  ctx.fillStyle='#91a5b8';ctx.font='13px sans-serif';ctx.fillText('Images are host references; stored state is connections.',x,698);
  const stored=new Set(rec.connections),added=new Set(delta?.added||[]);for(let bit=0;bit<1024;bit++){ctx.fillStyle=stored.has(bit)?added.has(bit)?'#7be1c0':'#e1bf7b':'#182a3b';ctx.fillRect(x+bit%32*4,727+Math.floor(bit/32)*4,3,3);}
  ctx.fillStyle='#e7edf5';ctx.font='18px monospace';ctx.fillText(`Stored VIS ${String(record).padStart(3,'0')}`,x+148,752);
  ctx.fillStyle='#c8d5e4';ctx.font='15px sans-serif';ctx.fillText('64 / 1024 stored input bits',x+148,783);
  ctx.fillText(`Captured score: ${recordScore(bank,record)}/64`,x+148,813);
  ctx.fillStyle='#7be1c0';ctx.font='16px monospace';ctx.fillText(`+${delta?.added.length||0} / −${delta?.removed.length||0} edits`,x+148,844);
  ctx.fillStyle='#91a5b8';ctx.font='13px sans-serif';ctx.fillText(`${summary.updated}/200 records have observed RAM updates.`,x,889);
  ctx.fillStyle='#26374b';ctx.fillRect(54,938,1812,4);ctx.fillStyle='#e1bf7b';ctx.fillRect(54,938,1812*(index-260)/139,4);
  ctx.fillStyle='#c8d5e4';ctx.font='18px sans-serif';ctx.fillText('Violet arrows: captured neuron output score. Gold/coral: derived gain/loss. Green: captured connection edits.',54,987);
  ctx.fillStyle='#91a5b8';ctx.font='16px sans-serif';ctx.fillText('Output arrows illustrate returned scores, not stored synapses. One learning NPU; logical geometry; editorial event cadence. No electrical spike timing measured.',54,1032);
  if(index>=399&&!clip.ending){clip.ending=true;stopPlayback();setTimeout(stopRecording,600);}
}
function drawRecording(t){
  if(!clip)return;
  const {ctx,c,start}=clip, e=event();if(!e)return;
  if(isRGB()){
    drawRGBRecording();if(clip.lastEvent!==index){clip.events.push({event:index,seconds:(performance.now()-start)/1000,role:e.role,image_id:e.image_id,counts:e.counts});clip.lastEvent=index;}return;
  }
  ctx.fillStyle='#080e17';ctx.fillRect(0,0,1920,1080);
  ctx.fillStyle='#e7edf5';ctx.font='500 34px sans-serif';ctx.fillText('AKD1500 / Network Explorer',54,60);
  ctx.fillStyle='#7be1c0';ctx.font='18px sans-serif';ctx.fillText(`LIVE SESSION · captured event ${index+1} · ${((performance.now()-start)/1000).toFixed(1)}s`,54,96);
  const sw=renderer.domElement.width,sh=renderer.domElement.height,scale=Math.min(1400/sw,780/sh),rw=sw*scale,rh=sh*scale,ox=35+(1400-rw)/2,oy=126+(780-rh)/2;
  const cw=Math.min(sw,sh*1400/780),cx=(sw-cw)/2;
  ctx.drawImage(renderer.domElement,cx,0,cw,sh,35,126,1400,780);
  for(const {el,position} of labels){
    const p=position.clone().project(camera);if(p.z>1||p.z< -1)continue;
    ctx.fillStyle=el.style.color||'#91a5b8';ctx.font=(el.querySelector('strong')?'600 22px':'15px')+' sans-serif';ctx.textAlign='center';
    ctx.fillText(el.textContent,35+(((p.x+1)/2*sw-cx)/cw)*1400,126+(-p.y+1)/2*780);
  }
  ctx.textAlign='left';const x=1470;
  ctx.fillStyle='#91a5b8';ctx.font='16px sans-serif';ctx.fillText('SELECTED LEARNING RECORD',x,160);
  ctx.fillStyle=colors[bank];ctx.font='26px sans-serif';ctx.fillText(`${bankNames[bank]} / ${record}`,x,202);
  ctx.fillStyle='#e7edf5';ctx.font='17px sans-serif';ctx.fillText(`NPU (${data.banks[bank].mapping.join(',')})`,x,232);
  const ss=stateAt(side==='before'),before=stateAt(true),rec=ss?.[bank][record],old=before?.[bank][record],delta=selectedChange();
  if(rec){
    ctx.fillStyle='#91a5b8';ctx.fillText(`${rec.connections.length} STORED CONNECTIONS · ${side.toUpperCase()}`,x,290);
    for(let j=0;j<rec.connections.length;j++){
      const id=rec.connections[j],cx=x+(j%4)*94,cy=325+Math.floor(j/4)*43;
      ctx.fillStyle=delta?.added.includes(id)?'#153e36':'#152334';ctx.fillRect(cx,cy-22,82,32);
      ctx.fillStyle=delta?.added.includes(id)?'#7be1c0':'#c8d5e4';ctx.fillText(String(id).padStart(3,'0'),cx+13,cy);
    }
    ctx.fillStyle='#7be1c0';ctx.fillText(`Added ${delta?.added.length||0} / removed ${delta?.removed.length||0}`,x,698);
    ctx.fillStyle='#91a5b8';ctx.fillText(`Reconstructed overlap ${e.derived_potentials[bank][record]} / ${bankBudget()}`,x,736);
    ctx.fillStyle='#c8d5e4';ctx.font='14px monospace';rec.words.forEach((v,j)=>ctx.fillText(`word ${j}  0x${v.toString(16).padStart(8,'0')}`,x,782+j*26));
  }
  ctx.strokeStyle='#26384a';ctx.beginPath();ctx.moveTo(54,925);ctx.lineTo(1866,925);ctx.stroke();
  ctx.font='22px sans-serif';ctx.fillStyle='#e7edf5';ctx.fillText(`${pretty(e.role)} · fact ${e.key} · source/context ${e.policy_key}`,54,964);
  ctx.font='18px sans-serif';ctx.fillStyle='#91a5b8';ctx.fillText(`Returned admission: ${$('decision').textContent}    Memory: ${$('memoryAnswer').textContent}`,54,1000);
  ctx.fillStyle='#c8d5e4';ctx.fillText(e.counts.map((v,k)=>`${bankNames[k]} +${v.added}/−${v.removed}`).join('     '),980,964);
  ctx.fillStyle='#91a5b8';ctx.font='15px sans-serif';ctx.fillText('Connections and returned flags measured. Glow / overlap reconstructed. Logical layout; host paced. CPU controls also pass.',54,1047);
  if(clip.lastEvent!==index){clip.events.push({event:index,seconds:(performance.now()-start)/1000,role:e.role,counts:e.counts});clip.lastEvent=index;}
}
function stopRecording(){if(clip&&clip.rec.state==='recording')clip.rec.stop();}
async function startRecording(){
  if(clip){stopRecording();return;}
  if(preparingClip)return;
  if(!isRGB()&&(!usingLive()||live.status!=='ready')){error('Start a live native trial before recording. Replays are not labeled live.');return;}
  completedClipURLs.forEach(u=>URL.revokeObjectURL(u));completedClipURLs=[];$('recordingDownloads').hidden=true;
  const imageCache=new Map();
  if(isRGB()){
    preparingClip=true;$('recordClip').disabled=true;$('recordClip').textContent='Preparing replay…';
    const requiredImages=new Set(events().slice(260,400).map(e=>e.image_id));for(const g of memorySummary(data,currentArm(),260,'after').groups)if(g.reference)requiredImages.add(g.reference);
    try{await Promise.all([...requiredImages].map(async id=>{const im=new Image();im.src=data.images[id].data_url;await im.decode();imageCache.set(id,im);}));}
    catch(e){error(e);preparingClip=false;$('recordClip').disabled=false;$('recordClip').textContent='Record replay clip';return;}
    preparingClip=false;$('recordClip').disabled=false;side='after';selectIndex(260);$('speed').value='8';playing=true;lastTick=performance.now();playbackLabels();
  }
  const c=document.createElement('canvas');c.width=1920;c.height=1080;
  const type=['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm'].find(v=>MediaRecorder.isTypeSupported(v));
  if(!type){error('This browser does not support WebM recording.');return;}
  const stream=c.captureStream(60),rec=new MediaRecorder(stream,{mimeType:type,videoBitsPerSecond:14000000});
  const chunks=[],saved={started_utc:new Date().toISOString(),source:isRGB()?'read-only RGB profile; actual saved chip events replayed in order':'real-time explorer canvas during resident native trial',path:isRGB()?data.source.study:live.path,arm:isRGB()?arm:live.arm,data_sha256:status.export.data_sha256,evidence:isRGB()?'Measured images, native scores and connection edits; gain/geometry derived; no electrical spikes':'Measured connections / returned flags; reconstructed activity; logical geometry'};
  if(isRGB())saved.neuron_output_trace='Captured pre-update per-record match score; schematic return path, not a stored synapse; no animated transit, electrical timing unknown.';
  clip={c,ctx:c.getContext('2d'),rec,start:performance.now(),events:[],lastEvent:-1,previousRotate:controls.autoRotate,previousPosition:camera.position.clone(),previousTarget:controls.target.clone()};
  if(isRGB()){
    const rgbRenderer=new THREE.WebGLRenderer({antialias:true,alpha:false,preserveDrawingBuffer:true});rgbRenderer.setClearColor('#080e17',1);rgbRenderer.outputEncoding=THREE.sRGBEncoding;rgbRenderer.setSize(2060,1460,false);
    const rgbCamera=new THREE.PerspectiveCamera(38,1030/730,.1,120),box=new THREE.Box3().setFromObject(dynamic);labels.forEach(l=>box.expandByPoint(l.position));const center=box.getCenter(new THREE.Vector3()),direction=new THREE.Vector3(1,2.2,14).normalize(),corners=[];
    for(const x of [box.min.x,box.max.x])for(const y of [box.min.y,box.max.y])for(const z of [box.min.z,box.max.z])corners.push(new THREE.Vector3(x,y,z));let distance=7;
    for(let i=0;i<50;i++){rgbCamera.position.copy(direction).multiplyScalar(distance).add(center);rgbCamera.lookAt(center);rgbCamera.updateMatrixWorld();if(corners.every(c=>{const p=c.clone().project(rgbCamera);return Math.abs(p.x)<.86&&Math.abs(p.y)<.82;}))break;distance*=1.1;}
    Object.assign(clip,{rgbRenderer,rgbCamera,center,direction,distance,imageCache});
  }else{camera.position.set(1.2,3.7,10.4);controls.target.set(0,1.1,0);controls.update();controls.autoRotate=false;}
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
  rec.onstop=()=>{
    const done=clip;clip=null;controls.autoRotate=done.previousRotate;camera.position.copy(done.previousPosition);controls.target.copy(done.previousTarget);controls.update();stream.getTracks().forEach(t=>t.stop());done.rgbRenderer?.dispose();
    const stamp=saved.started_utc.replaceAll(':','-'),download=(blob,name)=>{const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);};
    saved.events=done.events;saved.duration_seconds=(performance.now()-done.start)/1000;
    const prefix=isRGB()?'akd-rgb-recorded-explorer':'akd-live-explorer';
    if(isRGB()){
      const downloads=$('recordingDownloads');downloads.replaceChildren();
      for(const [blob,name,text] of [[new Blob(chunks,{type}),`${prefix}-${stamp}.webm`,'Save replay video'],[new Blob([JSON.stringify(saved,null,2)],{type:'application/json'}),`${prefix}-${stamp}.json`,'Save event receipt']]){const a=document.createElement('a');a.href=URL.createObjectURL(blob);completedClipURLs.push(a.href);a.download=name;a.textContent=text;downloads.append(a);}downloads.hidden=false;
    }else{
      download(new Blob(chunks,{type}),`${prefix}-${stamp}.webm`);
      download(new Blob([JSON.stringify(saved,null,2)],{type:'application/json'}),`${prefix}-${stamp}.json`);
    }
    $('recordClip').textContent=isRGB()?'Record replay clip':'Record live clip';liveButtons();
  };
  rec.start(1000);$('recordClip').textContent='Stop recording';if(isRGB())$('arm').disabled=true;setTimeout(stopRecording,60000);
}
$('recordClip').onclick=startRecording;
async function init(){
  [data,status]=await Promise.all([fetch('/api/replay').then(r=>r.json()),fetch('/api/status').then(r=>r.json())]);
  if(isRGB()){
    arm='adaptive';colors[0]='#e1bf7b';document.body.classList.add('visual-profile');document.querySelector('.sceneHint span').textContent='Links: captured input addresses. Input volume and spatial setting: schematic.';$('visualSourcePanel').hidden=false;$('links').value='selected';$('followEdits').parentElement.lastChild.textContent=' Follow response / edits';$('nodeConnections').parentElement.lastChild.textContent=' Selected inputs & neuron output';
    $('instrumentSubtitle').textContent='Recorded RGB → native memory; inspect every measured update';$('decisionHeading').textContent='OBSERVATION MODE';$('answerHeading').textContent='IMAGE IDENTITY';$('scoreHeading').textContent='CAPTURED NATIVE MATCH SCORE';
    $('scoreScope').textContent='The raw match score was returned by the chip before the update. Gain versus initial connections is derived. The image feature view is reconstructed; no electrical spike timing is measured.';
    $('comparisonScope').textContent='Held-out frames from one adapted recording. Initial old recall missed its 70% floor. CPU controls use the same features; no timing/energy comparison.';
    const evidence=document.querySelector('.evidence');evidence.querySelectorAll('p')[1].lastChild.textContent=' Captured native match scores';evidence.querySelectorAll('p')[2].lastChild.textContent=' Derived gain / reconstructed features';
    $('liveMode').title='No qualified live RGB manager actor. Use the separate three-bank profile for its existing live trial.';
  }
  if(isRepeated()){
    document.body.classList.add('visual-profile');$('visualSourcePanel').hidden=false;
    $('instrumentSubtitle').textContent='Repeated native replacement · recorded real-image episodes';
    $('decisionHeading').textContent='LEARNED NATIVE CONTROL';$('answerHeading').textContent='CAPTURED CONTENT RETURN';
    $('scoreHeading').textContent='RECONSTRUCTED MATCH / NATIVE RESPONSE';$('scoreScope').textContent='Binary responses are returned by the chip. Overlap is reconstructed from actual input and pre-update RAM. Geometry is schematic; no electrical spike timing is measured.';
    $('comparisonScope').textContent='Matched separate sessions. Prior-setting final recall alone hides its failed initial full-capacity check. Adaptive CPU stores also pass; no timing/energy claim.';
    $('recordClip').disabled=true;$('recordClip').title='Deterministic measured capture is available through this profile’s local renderer.';
    document.querySelector('.rail .fine').textContent='Fixed images and initialization; causal controls. Prior threshold setting also changes the fill behavior.';
  }
  $('dataset').value=isRepeated()?'8898':isRGB()?'8895':data.banks.length===4?'8893':'8892';$('dataset').disabled=true;$('dataset').title='This server serves one verified capture. Use --replay to choose another recorded profile.';
  bankNames=data.banks.map(b=>b.name);const legend=$('legend');legend.replaceChildren(...data.banks.map((b,k)=>{const sp=document.createElement('span'),sw=document.createElement('i');sw.className='swatch';sw.style.background=colors[k];sp.append(sw,document.createTextNode(b.name));return sp;}));const note=document.createElement('span');note.textContent=isRGB()?'Cyan volume: derived input bits · gold/coral glow: derived gain/loss · green/red links: captured edits · violet arrows: captured neuron response · annotation route is schematic; transit timing unknown':'Green/red: captured additions/removals · brightness: '+(isRepeated()?'actual returned binary response':'reconstructed output')+'';legend.append(note);if(isRGB()){$('recordClip').textContent='Record replay clip';$('recordClip').title='Record a read-only replay of captured RGB events; never opens the device.';}else if(!status.native_enabled){$('recordClip').disabled=true;$('recordClip').title='This dataset is a read-only recorded study; live capture stays on the qualified three-bank console.';}$('bankTotals').textContent=`${data.banks.reduce((n,b)=>n+b.plastic.length,0)} / ${data.banks.reduce((n,b)=>n+b.units,0)}`;$('liveMode').disabled=!status.native_enabled;
  for(const [key,value] of Object.entries(data.arms)){const o=document.createElement('option');o.value=key;o.textContent=value.label;$('arm').append(o);}
  for(let k=0;k<6;k++){const o=document.createElement('option');o.value=k;o.textContent=`Fact ${k}`;$('fact').append(o);}
  const stages=['Input encoder','Manager','Admission route','Memory encoder','Memory 0','Memory bridge','Memory 1 / readout'];
  data.mapping.forEach((m,k)=>{if(k){const arrow=document.createElement('span');arrow.className='routeArrow';arrow.textContent='→';$('routing').append(arrow);}const el=document.createElement('div');el.className='routeStage';const b=document.createElement('b');b.textContent=pretty(m.layer||stages[k]||'fixed stage');const s=document.createElement('small');s.textContent=m.nps.length===1?`NPU (${m.nps[0].np.join(',')})`:`${m.nps.length} mapped NPs`;s.title=m.nps.map(n=>n.np.join('.')).join(' · ');el.append(b,s);$('routing').append(el);});
  $('provenance').textContent=isRepeated()?data.source.study+'. Descriptor SHA256 '+data.source.descriptor_sha256+'. '+data.limits.join('. '):isRGB()?`${data.source.study}. Saved-raw audit SHA-256 ${data.source.mac_audit_sha256}. ${data.audit.images} images, ${data.audit.events_per_arm} measured events per arm. ${data.limits.join('. ')}.`:`${data.source.study}. Archive SHA-256 ${data.source.archive_sha256}. ${data.audit.independently_checked_streams} audited native/packed-CPU streams. ${data.limits.join('. ')}.`;
  $('sourceBadge').textContent=`SOURCE VERIFIED · ${Object.keys(data.arms).length} HARDWARE ARMS`;comparison();
  const initialEvents=data.arms[arm].events;index=isRepeated()?initialEvents.findIndex(e=>e.role==='retire'&&e.changes.some(c=>c.bank===1)):isRGB()?initialEvents.findIndex(e=>e.phase==='adapt'&&e.changes.length):initialEvents.findIndex(e=>e.phase==='changed_feedback_1'&&e.changes.length);if(index<0)index=initialEvents.findIndex(e=>e.role==='store'&&e.changes.length);if(index<0)index=initialEvents.findIndex(e=>e.changes.length);if(index<0)index=0;const change=event().changes[0];bank=change?.bank||0;record=change?.record||0;
  live=status.live;render();resize();fitView();liveButtons();
  const presentation=new URLSearchParams(location.search),requestedOpacity=presentation.get('background');
  for(const [key,max,setter] of [['event',events().length-1,v=>index=v],['bank',data.banks.length-1,v=>{bank=v;record=data.banks[bank].plastic[0];}],['record',65535,v=>{if(v<data.banks[bank].units)record=v;}]]){const text=presentation.get(key);if(text!==null&&/^\d{1,6}$/.test(text)&&Number(text)<=max)setter(Number(text));}if(presentation.has('event')||presentation.has('bank')||presentation.has('record'))render();
  if(requestedOpacity!==null&&/^\d{1,3}$/.test(requestedOpacity)&&Number(requestedOpacity)<=100)setBackground(requestedOpacity);
  if(presentation.get('view')==='nodes'){
    $('nodeConnections').checked=presentation.get('connections')==='1';$('nodeLabels').checked=presentation.get('labels')==='1';
    document.body.classList.toggle('node-labels',$('nodeLabels').checked);setNodeView(true);
  }
  window.akdRepeatedExplorer=isRepeated()?{ready:true,data,armTag:()=>arm,select:selectIndex,event,events,stateAt,renderer,scene,camera,controls,dynamic:()=>dynamic,labels:()=>labels,setSelection:(b,r)=>{bank=b;record=r;},render,stop:stopPlayback}:null;
  if(isRepeated())await installRepeatedFilm(window.akdRepeatedExplorer);
  requestAnimationFrame(animate);if(!isRGB()&&!isRepeated())setInterval(poll,1000);
}
init().catch(error);
