"""Bounded read-only projection of already verified explorer data for Tab5."""
from __future__ import annotations
import time


def compact(data, live, query, *, received_at=None, research_busy=False, schema=1):
    if schema not in (1,2):raise ValueError('Unsupported tablet schema')
    dimensions=data['banks'];bank_count=len(dimensions)
    if schema==1 and bank_count!=3:raise ValueError('This dataset requires /api/tab5/v2')
    if not 1<=bank_count<=4:raise ValueError('Unsupported bank count')
    for b in dimensions:
        width=b.get('inputs',512);budget=b.get('num_weights',32)
        if width not in (128,512) or budget not in (8,32) or not 1<=b['units']<=128:
            raise ValueError('Unsupported bank dimensions')
        if b['classes'] not in (2,4) or b['units']%b['classes']:
            raise ValueError('Unsupported class partition')
        if schema==1 and (width!=512 or budget!=32):raise ValueError('This dataset requires /api/tab5/v2')
    allowed={'mode','event','bank','record','arm'}
    if set(query)-allowed or any(len(v)!=1 for v in query.values()):raise ValueError('Unknown or repeated selector')
    mode=query.get('mode',['replay'])[0]
    if mode not in ['replay','live']:raise ValueError('mode must be replay or live')
    arm=query.get('arm',['native'])[0]
    if arm not in data['arms']:raise ValueError('Unknown arm')
    # The qualified live actor is still the original three-bank graph. A four-
    # bank recording must never supply its dimensions to that different stream.
    if mode=='live' and bank_count!=3:
        return dict(schema=schema,mode='live',status='unsupported_live_graph',available=False,
            controls_enabled=False,research_busy=bool(research_busy),banks=[],event=None)
    stream=live if mode=='live' else data['arms'][arm]
    es=stream.get('events') or []
    if mode=='live' and (not stream.get('initial') or not es):
        return dict(schema=schema,mode='live',status=stream.get('status','idle'),available=False,controls_enabled=False,research_busy=bool(research_busy),banks=[],event=None)
    def integer(key,default,lo,hi):
        raw=query.get(key,[str(default)])[0]
        if not raw.isdecimal() or not lo<=int(raw)<=hi:raise ValueError('Out of range '+key)
        return int(raw)
    index=integer('event',len(es)-1,0,len(es)-1);bank=integer('bank',0,0,bank_count-1)
    dim=data['banks'][bank];record=integer('record',dim['plastic'][0],0,dim['units']-1)
    states=[list(b) for b in stream['initial']];before=None
    for e in es[:index+1]:
        before=[list(b) for b in states]
        for c in e['changes']:states[c['bank']][c['record']]=c['after']
    e=es[index];change=next((c for c in e['changes'] if c['bank']==bank and c['record']==record),None)
    banks=[]
    for k,b in enumerate(data['banks']):
        potentials=e['derived_potentials'][k]
        banks.append(dict(id=k,name=b['name'],mapping=b['mapping'],units=b['units'],plastic_count=len(b['plastic']),
            edits=e['counts'][k],records=[dict(id=j,overlap=potentials[j],output=int(potentials[j]>b['thresholds'][j]),
                changed=any(c['bank']==k and c['record']==j for c in e['changes'])) for j in b['plastic']]))
        if schema==2:
            banks[-1].update(inputs=b.get('inputs',512),num_weights=b.get('num_weights',32),classes=b['classes'])
            for record_summary in banks[-1]['records']:
                record_summary['class_id']=record_summary['id']//(b['units']//b['classes'])
    age=None if mode=='replay' or received_at is None else round(max(0,time.time()-received_at),1)
    result=dict(schema=schema,mode=mode,arm=stream.get('arm',arm),status=stream.get('status','recorded'),available=True,
        controls_enabled=False,research_busy=bool(research_busy),age_seconds=age,event_count=len(es),
        event=dict(index=index,role=e['role'],phase=e['phase'],key=e['key'],policy_key=e['policy_key'],candidate=e['candidate'],outcome=e.get('outcome'),
                   policy_flags=e['policy_flags'],memory_flags=e['memory_flags'],counts=e['counts']),banks=banks,
        selection=dict(bank=bank,record=record,before=before[bank][record],after=states[bank][record],
                       added=change['added'] if change else [],removed=change['removed'] if change else [],
                       overlap=e['derived_potentials'][bank][record],threshold=dim['thresholds'][record]),
        evidence=dict(ram='measured',returned_flags='measured',overlap_and_unit_output='reconstructed',geometry='logical schematic',timing='event order, not electrical spike time'))
    if schema==2:
        result['study']=data.get('source',{}).get('study','20261025-learned-policy')
        result['event']['trace_flags']=e.get('trace_flags',[])
        result['selection'].update(inputs=dim.get('inputs',512),num_weights=dim.get('num_weights',32),
            class_id=record//(dim['units']//dim['classes']))
    return result
