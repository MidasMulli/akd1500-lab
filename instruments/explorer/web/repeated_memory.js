// Data binding for recorded native image-episode replacement. No device commands.
const $=id=>document.getElementById(id);
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;};
export function matchingKey(data,rec){
 const stored=new Set(rec.connections);
 return data.keys.findIndex(key=>key.every((v,j)=>Boolean(v)===stored.has(4*j))&&rec.connections.every(j=>j%4===0));
}
export function renderRepeatedComparison(root,data){
 const table=el('table');table.innerHTML='<thead><tr><th>Arm</th><th>Initial</th><th>Reused</th><th>Delayed</th></tr></thead>';const body=el('tbody');
 const rows=Object.values(data.arms).map(a=>[a.label,a.assessment.initial_recall,[a.assessment.cycles_detail.filter(c=>c.new_recall&&c.new_reuses_owner).length,4],a.assessment.delayed_recall]);
 for(const k of ['exact','nearest','fifo','lru']){const c=data.cpu_controls[k];rows.push(['CPU '+k,c.initial,c.new,c.delayed]);}
 for(const row of rows){const tr=el('tr');row.forEach((v,j)=>tr.append(el('td',null,j?v.join('/'):v)));body.append(tr);}table.append(body);root.replaceChildren(table);
}
export function renderRepeatedEvent({data,arm,e,state,before,side,bank,record,onSelect}){
 const titles={bootstrap:'Learn the usefulness rule',fill:'Fill eight native memories',before:'Verify a full memory pool',reverse:'The usefulness rule changes',retire:'Retire a matching old episode',unknown_bad:'Reject an unfamiliar, unhelpful episode',new_key:'Learn the replacement',delay:'Intervening image observations',delayed:'Delayed recall',context_check:'Check content across both contexts'};
 $('phaseTitle').textContent=(e.cycle>=0?'Cycle '+(e.cycle+1)+' · ':'')+(titles[e.role]||e.role);
 const image=e.image_id?data.images[e.image_id]:null;$('visualImage').hidden=!image;
 if(image){$('visualImage').src=image.data_url;$('visualImage').alt='Recorded image episode '+e.key;$('visualImageMeta').textContent='CORe50 · object '+image.object+' · session '+image.session+' · frame '+image.frame;}
 else $('visualImageMeta').textContent='Context teaching · no image is supplied in this event';
 $('visualMode').textContent=e.teacher_enabled?'NATIVE LEARNING ENABLED':'READ ONLY';$('visualMode').classList.toggle('teaching',e.teacher_enabled);
 $('visualTarget').textContent=image?'Image episode E'+e.key:'Teach context '+e.quality+' as '+(e.teaching_label===0?'useful':'not useful');
 $('visualExplanation').textContent='Frozen host CNN → binary descriptor. Native policy and content memory update separately.';
 const metrics=arm.assessment;
 document.querySelector('.visualSummary').replaceChildren(el('div','eyebrow','COMPLETE RECORDING'),el('p',null,metrics.delayed_recall.join('/')+' delayed recalls · '+metrics.new_recall.join('/')+' new episodes'),el('p','fine','Four cycles · 16 other image observations before each delayed exam.'));
 $('eventDescription').textContent=(e.active_bank===0?'Control':'Content')+' · '+(image?'E'+e.key:'context teacher');
 $('eventDetail').textContent='Six co-resident FNP3s; host-paced role/configuration clocks. This is recorded chip data.';
 const flags=e.control_flags;$('decision').textContent=flags[1]?'USEFUL':flags[2]?'NOT USEFUL':'NO POLICY RETURN';
 $('decision').style.color=flags[1]?'#7be1c0':'#e1bf7b';$('decisionDetail').textContent='Last native control: recall feedback '+flags[0]+' · retirement permitted '+flags[3];
 const response=e.native_responses[1];
 $('memoryAnswer').textContent=response?(e.returned_record===null?'NO EXACT MATCH':e.recalled_key===null?'Record M'+e.returned_record:'Episode E'+e.recalled_key):'RETURN PENDING';
 $('memoryAnswer').style.color=response&&e.returned_record!==null?'#7be1c0':'#91a5b8';
 $('cpuAnswer').textContent=response?'Actual eight-bit return ['+response.join(', ')+']; before this step’s update.':'Control role executed; no fresh content return is attributed to this event.';
 const root=$('visualMemoryPanel');root.hidden=false;root.classList.add('repeated-memory');root.replaceChildren();
 root.append(el('div','eyebrow','NATIVE CONTENT RETURN'));
 root.append(el('p',null,response?response.join('  '):'Awaiting content role'));
 root.append(el('div','eyebrow memoryHeading','STORED DESCRIPTORS · IMAGE REFERENCES'));
 const gallery=el('div','memoryGallery');
 state[1].forEach((rec,j)=>{
  const key=matchingKey(data,rec),change=e.changes.find(c=>c.bank===1&&c.record===j),hot=response?.[j]===1;
  const card=el('button','memoryCard'+(bank===1&&j===record?' selected':'')+(hot?' recalled':''));card.type='button';card.onclick=()=>onSelect(j);
  if(key>=0){const im=el('img');im.src=data.images['episode'+key].data_url;im.alt='Host image reference matching captured descriptor E'+key;card.append(im);}
  else card.append(el('span','emptyReference',rec.words[3]>>>16===0?'Unused':rec.words[3]>>>16===1?'Reusable':'Updating'));
  card.append(el('span',null,'M'+j+(key>=0?' · E'+key:'')),el('small',null,'Learn >'+(rec.words[3]>>>16)+(change?' · RAM edit':'')));gallery.append(card);
 });
 root.append(gallery,el('p','memoryScope','Thumbnails identify exact captured descriptor matches. The AKD stores connections, not RGB pixels. '+side+' event '+(e.index+1)+'.'));
 const active=e.native_responses[bank],v=active?.[record];
 $('unitOutput').textContent=e.active_bank!==bank?'This role was not executed at this event. Stored RAM remains inspectable.':v===null?'Helper overlap is reconstructed; this helper has no direct returned-output tap.':'Captured binary output: '+v+'. Match overlap below is reconstructed from the actual input and pre-update RAM.';
}
