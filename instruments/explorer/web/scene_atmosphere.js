// Schematic setting and derived input-volume rendering. No hardware or data edits.
import * as THREE from './vendor/three.module.js';

export function featurePosition(id){
  const bit=id%512,a=bit*2.399963229728653,r=.2+.83*Math.sqrt((bit%109)/109),branch=id<512?-1:1;
  return new THREE.Vector3(Math.cos(a)*r+branch*.13,.48+(Math.floor(bit/64)-3.5)*.2+Math.sin(a)*.09,Math.sin(a)*.88+branch*.12);
}
function radialTexture(){
  const c=document.createElement('canvas');c.width=c.height=64;const ctx=c.getContext('2d'),g=ctx.createRadialGradient(32,32,1,32,32,32);
  g.addColorStop(0,'rgba(255,255,255,.8)');g.addColorStop(.2,'rgba(255,255,255,.35)');g.addColorStop(.55,'rgba(255,255,255,.06)');g.addColorStop(1,'rgba(255,255,255,0)');ctx.fillStyle=g;ctx.fillRect(0,0,64,64);return new THREE.CanvasTexture(c);
}
const haloTexture=radialTexture();
export function responseHalo(position,color,strength,edited=false){
  if(strength<.08&&!edited)return null;
  const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:haloTexture,color,transparent:true,opacity:Math.min(.5,.14+strength*.3),depthWrite:false,blending:THREE.AdditiveBlending}));
  sprite.position.copy(position);sprite.scale.setScalar(.35+strength*.62);sprite.userData.shared=true;return sprite;
}
export function inputVolume(active,count,highlight=null){
  const pos=new Float32Array(count*3),color=new Float32Array(count*3),on=new THREE.Color('#76c8e8'),off=new THREE.Color('#152b40'),white=new THREE.Color('#eef7ff');
  for(let id=0;id<count;id++){const p=featurePosition(id);pos.set([p.x,p.y,p.z],id*3);const c=id===highlight?white:active.has(id)?on:off;color.set([c.r,c.g,c.b],id*3);}
  const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setAttribute('color',new THREE.BufferAttribute(color,3));
  const points=new THREE.Points(geo,new THREE.PointsMaterial({size:.28,map:haloTexture,vertexColors:true,transparent:true,opacity:.9,depthWrite:false,blending:THREE.AdditiveBlending}));
  points.userData.kind='derived-input-volume';return points;
}
export function memorySetting(){
  // Neutral group boundary for legibility; no die, board, routing or timing claim.
  const group=new THREE.Group();group.name='logical-bank-backdrop';
  const panel=new THREE.Mesh(new THREE.PlaneGeometry(10.3,6.25),new THREE.ShaderMaterial({
    vertexShader:'varying vec2 v;void main(){v=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
    fragmentShader:'varying vec2 v;void main(){vec2 q=(v-.5)*vec2(1.,1.2);float f=exp(-dot(q,q)*6.);gl_FragColor=vec4(mix(vec3(.022,.037,.057),vec3(.045,.096,.145),f),1.);}'
  }));panel.position.set(0,1,-1.93);group.add(panel);
  const border=new THREE.LineSegments(new THREE.EdgesGeometry(panel.geometry),new THREE.LineBasicMaterial({color:'#294058',transparent:true,opacity:.4}));border.position.copy(panel.position);group.add(border);
  const key=new THREE.PointLight('#6cb8e1',.65,18,2);key.position.set(-3,4,4);group.add(key);const warm=new THREE.PointLight('#e6c48a',.35,14,2);warm.position.set(4,1,3);group.add(warm);
  group.visible=false;return group;
}
