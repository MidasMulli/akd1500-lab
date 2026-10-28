import * as THREE from './vendor/three.module.js';

// This profile returns integer match scores. No threshold/spike flag is inferred.
export function neuronResponse(data,event,record){
  const bank=data.banks[0],value=event.observed_potentials[0][record];
  if(data.profile!=='rgb-native-memory'||bank.output_kind!=='raw_scores'||!Number.isInteger(value)||value<0||value>bank.num_weights)throw Error('Expected a captured native per-record match score');
  return {record,value,max:bank.num_weights,kind:'captured_match_score',preUpdate:true,usedByClassReadout:bank.plastic.includes(record)};
}
const arrowGeometry=new THREE.ConeGeometry(.045,.15,12);
const color='#c5a9ff';
export function responseTrace(from,response){
  const group=new THREE.Group(),endpoint=new THREE.Vector3(3.25,2.65,1.25),tip=endpoint.clone().add(new THREE.Vector3(.94,-.05,.04));
  group.name='schematic-returned-score';
  const path=new THREE.CubicBezierCurve3(from.clone(),from.clone().add(new THREE.Vector3(.5,.2,1.1)),new THREE.Vector3(4.6,1.9,1.8),tip);
  const line=new THREE.Mesh(new THREE.TubeGeometry(path,64,.022,6,false),new THREE.ShaderMaterial({
    uniforms:{tint:{value:new THREE.Color(color)}},transparent:true,depthWrite:false,
    vertexShader:'varying vec2 v;void main(){v=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
    fragmentShader:'uniform vec3 tint;varying vec2 v;void main(){if(fract(v.x*14.)>.64)discard;gl_FragColor=vec4(tint,.95);}'
  }));group.add(line);
  for(const t of [.55,.96]){const arrow=new THREE.Mesh(arrowGeometry,new THREE.MeshBasicMaterial({color,transparent:true,opacity:.95}));arrow.userData.shared=true;arrow.position.copy(path.getPoint(t));arrow.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),path.getTangent(t).normalize());group.add(arrow);}
  const box=new THREE.Mesh(new THREE.BoxGeometry(1.88,.72,.055),new THREE.MeshBasicMaterial({color:'#1c1731',transparent:true,opacity:.86}));box.position.copy(endpoint);group.add(box);
  const outline=new THREE.LineSegments(new THREE.EdgesGeometry(box.geometry),new THREE.LineBasicMaterial({color:'#8c70b6',transparent:true,opacity:.8}));outline.position.copy(endpoint);group.add(outline);
  return {group,path,endpoint,response};
}
