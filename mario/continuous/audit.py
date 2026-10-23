"""Independently replay each captured frame and bind native decisions to actual short outcomes."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tools'))
from verify import verify,read_words
from mario.sealed.bank import encode,save,sha
from mario.continuous.environment import contexts,opened,observation,button


def audit(study,capture,raw):
    report=json.loads((capture/'REPORT.json').read_text());assert report['physical'] and report['constant_override'] is None
    assert sha(capture/'events.jsonl')==report['events_sha256'] and sha(capture/'game.npz')==report['game_sha256']
    native=verify(ROOT,raw);assert native['samples']==report['forwards']
    events=[json.loads(line) for line in (capture/'events.jsonl').read_text().splitlines()]
    forwards=[e for e in events if e['kind']=='forward'];packets={r['id']:r for r in map(json.loads,(raw/'native.stdout').read_text().splitlines()) if r['kind']=='forward'}
    inputs=np.fromfile(raw/'fixture.inputs','u1').reshape(-1,1024);labels=np.fromfile(raw/'fixture.labels','<u4');score=np.fromfile(raw/'native.scores','<u4').reshape(-1,192)
    original=read_words(raw/'fixture.ram');before=original;queries=edits=0
    assert len(forwards)==len(inputs)==len(packets)
    for i,e in enumerate(forwards):
        r=e['response'];assert r['index']==i and r['id']==i+1
        assert all(r[k]==v for k,v in packets[r['id']].items())
        assert r['scores']==score[i].tolist() and r['label']==labels[i] and r['input_sha256']==hashlib.sha256(inputs[i].tobytes()).hexdigest()
        assert r['sent_monotonic_ns']<=r['received_monotonic_ns']
        after=read_words(raw/f'native.snapshot_{i}.ram')
        assert after[2+128*36:]==original[2+128*36:]
        changed=[j for j in range(192) if before[2+36*j:2+36*(j+1)]!=after[2+36*j:2+36*(j+1)]]
        bits=sum((a^b).bit_count() for j in changed for a,b in zip(before[2+36*j+4:2+36*(j+1)],after[2+36*j+4:2+36*(j+1)]))
        assert changed==r['changed_records'] and bits==r['connection_bit_edits'];edits+=bits
        if r['label']==2:assert before==after;queries+=1
        if report['arm']=='frozen':assert after==original
        before=after
    preparation=report['training_presentations'];fixture=study/'fixtures-loop01'/f's{report["seed"]}_{report["arm"]}'
    assert np.array_equal(inputs[:preparation],np.fromfile(fixture.with_suffix('.inputs'),'u1').reshape(-1,1024))
    assert np.array_equal(labels[:preparation],np.fromfile(fixture.with_suffix('.labels'),'<u4'))
    game=np.load(capture/'game.npz',allow_pickle=False);frames=game['frames'];clock=game['mono_offsets_ns'];assert np.all(np.diff(clock)>0)
    frame_events=[e for e in events if e['kind']=='frame'];assert [e['frame_id'] for e in frame_events]==list(range(len(frames)))
    origin=next(e['mono_ns'] for e in events if e['kind']=='game_start');assert all(e['mono_ns']==origin+int(clock[e['frame_id']]) for e in frame_events)
    replies={e['response']['id']:e for e in forwards};checked=set();decisions=0;online_edits=0
    for episode,c in enumerate(contexts(report['split'])):
        env,history,sky,info,valid=opened(c)
        try:
            assert valid
            es=[e for e in frame_events if e['episode']==episode];cursor=0
            assert np.array_equal(history[-1],frames[es[0]['frame_id']]);cursor+=1
            start_x=int(info['x_pos']);controlled=0;dead=flag=False
            ds=[e for e in events if e['kind']=='decision' and e['episode']==episode]
            os=[e for e in events if e['kind']=='outcome' and e['episode']==episode];assert len(ds)==len(os)
            for d,o in zip(ds,os):
                assert d['constant_override'] is None and d['controlled']==controlled
                assert d['input_frame']==es[cursor-1]['frame_id'] and d['elapsed']==c['wait']+c['start']+controlled
                expected=[encode(**observation(history,sky),action=a,encoder='projection',elapsed=d['elapsed']) for a in [0,1]]
                rr=[replies[r]['response'] for r in d['native_ids']]
                assert d['native_ids'][1]==d['native_ids'][0]+1
                for a,r in enumerate(rr):
                    assert r['label']==2 and replies[r['id']]['input_frame']==d['input_frame'] and np.array_equal(inputs[r['index']],expected[a]);checked.add(r['index'])
                margin=[max(r['scores'][64:128])-max(r['scores'][:64]) for r in rr]
                assert margin==d['margins'] and d['action']==int(margin[1]>margin[0])
                assert max(r['received_monotonic_ns'] for r in rr)<=d['mono_ns']
                previous_x=int(info['x_pos'])
                for k in range(min(24,1200-controlled)):
                    b=button(d['action'],k);frame,_,terminated,truncated,info=env.step(b);history.append(frame.copy());controlled+=1
                    flag=bool(info.get('flag_get'));dead=bool(info.get('death') or ((terminated or truncated) and not flag))
                    e=es[cursor];cursor+=1;assert e['button']==b and e['controlled']==controlled
                    assert np.array_equal(frame,frames[e['frame_id']]),('fresh game frame mismatch',episode,controlled)
                    assert e['mono_ns']>=d['mono_ns']
                    if dead or flag:break
                advance=int(info['x_pos'])-previous_x;success=bool(not dead and advance>=12)
                assert o['success']==success and o['advance_evaluation_only']==advance and o['dead']==dead and o['flag']==flag
                r=replies[o['response_id']]['response'];assert r['label']==(2 if report['arm']=='frozen' else int(success))
                assert np.array_equal(inputs[r['index']],expected[d['action']]) and replies[r['id']]['input_frame']==d['input_frame']
                assert r['sent_monotonic_ns']>=es[cursor-1]['mono_ns'];checked.add(r['index']);online_edits+=r['connection_bit_edits'];decisions+=1
            assert cursor==len(es) and (dead or flag or controlled==1200)
            row=report['episodes'][episode];assert row['advance_evaluation_only']==int(info['x_pos'])-start_x and row['controlled_frames']==controlled and row['decisions']==len(ds)
        finally:env.close()
    assert checked==set(range(preparation,len(inputs)))
    result=dict(status='INDEPENDENT_CONTINUOUS_AUDIT_PASS',native_forwards=len(inputs),frames=len(frames),repeated_decisions=decisions,
                protected_queries=queries,connection_bit_edits=edits,online_connection_replacements=online_edits//2,physical=True,
                all_RGB_frames_freshly_reproduced=True,all_live_inputs_reconstructed=True,source_sha256=sha(__file__))
    save(capture/'INDEPENDENT_AUDIT.json',result);print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',type=Path);p.add_argument('capture',type=Path);p.add_argument('raw',type=Path);a=p.parse_args();audit(a.study.resolve(),a.capture.resolve(),a.raw.resolve())
