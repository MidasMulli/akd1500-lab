"""Prospective fault-source swap; unchanged native circuit and packed simulator."""
import types
import numpy as np
from admission import consensus as circuit
from admission.consensus import N,D,DIMENSIONS,GROUPS,PT,PR,SEEDS,ROLES,weights,matrices,packed,thresholds
ARMS=['native'];STAGES=['crossed']

def schedule(stage,arm):
    assert stage=='crossed' and arm=='native'
    rows=[dict(e,labels=e['labels'].copy()) for e in circuit.schedule('repeat','native')[:728]]
    for e in rows:
        # Cycle one: B receives false evidence. Cycle two: A receives it.
        # Swap source identities for every ground role in the second cycle,
        # including restoration and legitimate updates; keep all fact values.
        if e['phase'].endswith('_2') and e['role'] in ['ground','ground_negative']:
            e['cohort']=1-e['cohort'];e['wire']^=2;e['labels'][0]=e['wire']
    assert len(rows)==728
    return rows

# Inject only the schedule into a separate globals dictionary; never mutate the
# historical module. Reuse its exact simulation bytecode and native circuit.
_simulate=types.FunctionType(circuit.simulate.__code__,dict(circuit.simulate.__globals__,schedule=schedule),name='crossed_packed_replay')

def simulate(rams,stage,arm):
    a=_simulate(rams,stage,arm);source_a={};answers=[]
    for e in a['events']:
        if e['role']=='ground' and e['cohort']==0:source_a[e['key']]=e['fact']
        # This CPU control copies A on a transfer, without seeing truth or phases.
        if e['index']==0:student={}
        if e['role']=='transfer' and e['key'] in source_a:student[e['key']]=source_a[e['key']]
        answers.append(student.get(e['key'],-1))
    a['cpu']=np.column_stack((a['cpu'],np.asarray(answers,'i4')))
    return a

def assess(a):
    out=circuit.assess(a)
    for phase,v in out.items():
        ii=[e['index'] for e in a['events'] if e['phase']==phase and e['role'] in ['query','exam']]
        v['cpu_follow_a_correct']=sum(int(a['cpu'][i,2]==a['events'][i]['actual']) for i in ii)
    return out
