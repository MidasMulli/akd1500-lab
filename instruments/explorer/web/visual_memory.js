// Read-only summaries of recorded native scores and captured memory boundaries.
export function memorySummary(data,arm,index,side='after'){
  const b=data.banks[0],e=arm.events[index],perClass=b.units/b.classes;
  const classes=Array.from(new Set(b.plastic.map(j=>Math.floor(j/perClass)))).sort((a,b)=>a-b);
  const updated=new Set(),references=new Map(),end=index+(side==='after'?1:0);
  for(let i=0;i<end;i++){
    const row=arm.events[i];if(row.teacher_enabled&&classes.includes(row.teaching_label)&&!references.has(row.teaching_label))references.set(row.teaching_label,row.image_id);
    for(const change of row.changes)if(change.bank===0&&b.plastic.includes(change.record))updated.add(change.record);
  }
  const groups=classes.map(id=>{
    const records=b.plastic.filter(j=>Math.floor(j/perClass)===id),score=Math.max(...records.map(j=>e.observed_potentials[0][j]));
    const matches=records.filter(j=>e.observed_potentials[0][j]===score);
    return {id,label:b.class_labels[id],records,score,bestRecord:matches[0],tiedRecords:matches.length,updated:records.filter(j=>updated.has(j)).length,reference:references.get(id)||null};
  });
  const best=Math.max(...groups.map(g=>g.score)),tied=groups.filter(g=>g.score===best),winner=groups.find(g=>g.id===e.prediction);
  if(!winner||winner.id!==tied[0].id)throw Error('Recorded readout disagrees with captured class-score argmax');
  return {groups,winner,tied,updated:updated.size,boundary:end,perClass};
}
const el=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;};
export function renderMemoryPanel(root,{data,arm,index,side,state,record,onSelect}){
  const s=memorySummary(data,arm,index,side),e=arm.events[index],winner=s.winner,b=data.banks[0];root.hidden=false;root.replaceChildren();
  root.append(el('div','eyebrow','READOUT · HOST FROM CHIP SCORES'));
  const result=el('button','readoutWinner'+(e.prediction===e.evaluation_target?'':' incorrect'));result.type='button';result.setAttribute('aria-label',`Inspect readout ${winner.label}, record ${winner.bestRecord}`);result.onclick=()=>onSelect(winner.bestRecord);
  if(winner.reference){const img=el('img');img.src=data.images[winner.reference].data_url;img.alt=`Host teaching reference for ${winner.label}`;result.append(img);}
  const heading=el('span','readoutHeading');heading.append(el('strong',null,winner.label),el('span',null,`${winner.score}/${b.num_weights} native match`),el('small',null,e.prediction===e.evaluation_target?'Correct for this image':'Incorrect for this image'));result.append(heading);root.append(result);
  const bars=el('div','classScoreGrid');bars.setAttribute('aria-label','Captured best match per object');
  for(const g of s.groups){const row=el('div','classScoreRow'+(g.id===winner.id?' winning':''));row.title=`${g.label}: ${g.score}/${b.num_weights}; ${g.tiedRecords} record(s) share this class maximum`;
    const track=el('span','scoreTrack'),fill=el('i');fill.style.width=`${g.score/b.num_weights*100}%`;track.append(fill);row.append(el('span',null,g.label),track,el('b',null,String(g.score)));bars.append(row);}
  root.append(bars);root.append(el('p','memoryScope',s.tied.length>1?`Tied classes: ${s.tied.map(g=>g.label).join(', ')}. Host argmax selects the first. Scores are pre-update matches, not probabilities.`:'Highest captured match per object; host argmax chooses the readout. Scores precede the update and are not probabilities.'));
  root.append(el('div','eyebrow memoryHeading','OBJECT MEMORY · CAPTURED STATE'));
  const gallery=el('div','memoryGallery');gallery.setAttribute('aria-label','Inspect object memory groups');
  const selectedClass=Math.floor(record/s.perClass);
  for(const g of s.groups){const card=el('button','memoryCard'+(g.id===selectedClass?' selected':'')+(g.id===winner.id?' recalled':''));card.type='button';card.setAttribute('aria-label',`Inspect ${g.label} memory, ${g.updated} of ${g.records.length} records updated`);card.title=`${g.label}: ${g.updated} records with observed RAM changes; ${g.score}/${b.num_weights} current best match. Thumbnail is a host teaching reference.`;card.onclick=()=>onSelect(g.bestRecord);
    if(g.reference){const im=el('img');im.src=data.images[g.reference].data_url;im.alt=`Host teaching reference for ${g.label}`;card.append(im);}else card.append(el('span','emptyReference','—'));
    card.append(el('span',null,g.label),el('small',null,`${g.updated}/${g.records.length} updated`));gallery.append(card);}
  root.append(gallery);root.append(el('p','memoryScope',`${side==='before'?'Before':'After'} event ${index+1}: ${s.updated}/${b.plastic.length} records have observed updates. Images are host teaching references; the chip stores connection patterns.`));
  const rec=state[0][record],sig=el('div','storedSignature'),canvas=el('canvas');canvas.width=canvas.height=160;canvas.setAttribute('aria-label',`Captured connection mask for record ${record}: ${rec.connections.length} of ${b.inputs} input bits stored`);
  const ctx=canvas.getContext('2d'),stored=new Set(rec.connections),added=new Set(side==='after'?(e.changes.find(c=>c.bank===0&&c.record===record)?.added||[]):[]);
  for(let i=0;i<b.inputs;i++){ctx.fillStyle=stored.has(i)?added.has(i)?'#7be1c0':'#e1bf7b':'#182a3b';ctx.fillRect(i%32*5,Math.floor(i/32)*5,4,4);}
  const info=el('div');info.append(el('b',null,`Stored record ${String(record).padStart(3,'0')}`),el('span',null,selectedClass<10?b.class_labels[selectedClass]:'Protected query helper'),el('span',null,`${rec.connections.length}/${b.inputs} connections`),el('small',null,'Captured SRAM mask · no RGB pixels'));
  sig.append(canvas,info);root.append(sig);
}
