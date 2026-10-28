"""Read-only export of actual C32 native taps/RAM; no device or training access."""
import hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np
from PIL import Image

B = Path(__file__).resolve().parents[1]
V = B / 'visual'
V.mkdir(exist_ok=True)
M = B / 'peer-media/MEDIA_R_DEV_K16'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
load = lambda p: json.loads(p.read_text())
lines = lambda p: [json.loads(x) for x in p.read_text().splitlines()]
assert sha(M/'SUMMARY.json') == 'eb66824bbbd71364a413faceb911060acc2ae1c7ba58e33cdb477f557a226934'
assert sha(M/'MANIFEST.jsonl') == 'f5a9df938cea5bc8b8b3272714794ea06c1921a9817964247347b9b5f18cb728'
manifest = lines(M/'MANIFEST.jsonl')
assert len(manifest) == 64 and load(M/'SUMMARY.json')['errors'] == 0
images, media = {}, {}
for m in manifest:
    assert not m['errors'] and m['checks']['query_bitwise_all_refs'] and m['checks']['feedback_match_all_refs']
    media[(m['scene_id'],m['program_id'],m['twin'],m['regime'])] = m
    for kind, info in m['images'].items():
        p = M/info['file']
        assert sha(p) == info['png_sha256']
        im = np.asarray(Image.open(p).convert('RGB'))
        assert list(im.shape) == info['shape'] and hashlib.sha256(im.tobytes()).hexdigest() == info['rgb_sha256']
        target = V/info['file']; target.parent.mkdir(exist_ok=True)
        shutil.copyfile(p,target)
        images[m['tag']+'_'+kind] = dict(src=info['file'],t=info['t'],png_sha256=info['png_sha256'])
audit = lines(B/'data-native02/AUDIT.jsonl')
query = lines(B/'data-native02/QUERY.jsonl')
feedback = lines(B/'data-native02/FEEDBACK.jsonl')
records = {c:lines(B/f'native-{c}/RECORDS.jsonl') for c in ['02','03','04']}
assert len(audit) == len(query) == len(records['02']) == 144
for c in records:
    assert load(B/f'native-{c}/TASK_QA.json')['status'] == 'PASS'
    assert [r['query_sha'] for r in records[c]] == [r['query_sha'] for r in records['02']]
for t,q,f in zip(audit,query,feedback):
    m = media[(t['scene_id'],t['program_id'],t['twin'],t['regime'])]
    assert m['checks']['query_sha_recomputed'] == q['query_sha']
    assert m['checks']['outcome'] == f['outcome'] and m['checks']['t_end'] == f['t_end']

# Layout constants copied from the frozen, qualified oct9 geometry, not fitted.
P, W, N1, U1 = 176,512,704,32
from types import SimpleNamespace
g = SimpleNamespace(**load(B/'native-02/fixture/LAYOUT.json'))
value = g.value
assert (g.P,g.W,g.n1,g.u1) == (P,W,N1,U1)
assert np.all(np.fromfile(B/'native-02/fixture/bank1.threshold','<i4')==31)
catalog, by_addresses = [], {}
for r,t in zip(records['02'],audit):
    key = np.frombuffer(bytes.fromhex(r['key_hex']),'u1').copy()
    assert len(key) == P and key.sum() == 31
    key[value:value+3] = 0; key[value+r['truth']] = 1
    addresses = tuple((4*np.flatnonzero(key)).tolist())
    if addresses not in by_addresses:
        m = media[(t['scene_id'],t['program_id'],t['twin'],t['regime'])]
        by_addresses[addresses] = len(catalog)
        catalog.append(dict(addresses=list(addresses),set=m['set'],context=m['context_index'],regime=t['regime'],outcome=r['truth'],first_episode=r['seq'],tag=m['tag']))
assert len(catalog) == 64
archive = B/'NATIVE_02.zip'
assert sha(archive) == 'bc2f0f99b1ad528e056af9d53ca5de879e596b689c0a22189013e5c44b7abe63'
z = zipfile.ZipFile(archive)
qa = json.loads(z.read('RAW.QA.json'))
assert qa['physical'] and qa['status'] == 'PASS'
rows = load(B/'native-02/fixture/ROWS.json'); n = len(rows)
out = np.frombuffer(z.read('RAW.outputs'),'<u4').reshape(n,W)
ram = np.frombuffer(z.read('RAW.states1'),'<u4').reshape(n,834)
inputs = np.frombuffer(z.read('RAW.actual_inputs1'),'u1').reshape(-1,N1)
initial = np.frombuffer(z.read('RAW.initial1'),'<u4')
versions, version_ids = [], {}
def memory(a):
    result=[]
    for j,w in enumerate(a[2:].reshape(U1,26)):
        addr=np.flatnonzero(np.unpackbits(w[4:].copy().view('u1'),bitorder='little')).tolist()
        assert len(addr)==32 and all(x%4==0 for x in addr)
        threshold=int(w[3]>>16)
        owner=by_addresses.get(tuple(addr)) if threshold>=30 else None
        result.append(dict(id=j,addresses=addr,learn_threshold=threshold,owner=owner))
    return result
def version(a):
    m=memory(a); token=json.dumps(m,separators=(',',':'))
    if token not in version_ids:
        version_ids[token]=len(versions);versions.append(m)
    return version_ids[token]
old=initial; before_id=version(old); events=[]; at=0; flags=[0,0,0]
last_net1={r['episode']:i for i,r in enumerate(rows) if r['net']==1 and r['episode']>=0}
edited_records=set(); direct_taps=0; total_edits=0
for i,r in enumerate(rows):
    if r['net']==0:
        flags=[int(out[i,P+g.error_probe]),int(out[i,P+g.correct_probe]),int(out[i,P+g.retire])]
        assert np.array_equal(old,ram[i]);continue
    x=inputs[at];at+=1
    before=memory(old); after_id=version(ram[i]);after=versions[after_id]
    response=out[i,1:33].astype(int).tolist()
    active=np.flatnonzero(x).tolist();active_set=set(active)
    overlaps=[len(active_set.intersection(m['addresses'])) for m in before]
    assert response==[int(v>31)for v in overlaps] and int(out[i,0])==int(any(response))
    direct_taps+=32
    changes=[]
    for j,(a,b) in enumerate(zip(before,after)):
        if a==b:continue
        add=sorted(set(b['addresses'])-set(a['addresses']));remove=sorted(set(a['addresses'])-set(b['addresses']))
        changes.append(dict(record=j,added=add,removed=remove,threshold_before=a['learn_threshold'],threshold_after=b['learn_threshold']))
        if add or remove:edited_records.add(j);total_edits+=len(add)
        if b['owner'] is not None:assert catalog[b['owner']]['first_episode']<=r['episode']
    old=ram[i]
    if r['episode']<0:before_id=after_id;continue
    # Every query, visible connection/threshold edit, and final feedback is kept.
    if r['role']=='query' or changes or last_net1[r['episode']]==i:
        ep=r['episode'];t=audit[ep];m=media[(t['scene_id'],t['program_id'],t['twin'],t['regime'])]
        rec=records['02'][ep]
        progress={}
        for cell,arm in [('02','error-only'),('03','all-admit'),('04','S1AB-frozen')]:
            done=[j for j in range(ep+int(r['role']=='feedback')) if audit[j]['phase']==3 and audit[j]['repeat']==1]
            progress[arm]=dict(n=len(done),correct=sum(int(np.argmax(records[cell][j]['predictions'][arm]))==records[cell][j]['truth'] for j in done))
        events.append(dict(index=i,episode=ep,phase=t['phase'],repeat=t['repeat'],set=m['set'],context=m['context_index'],tag=m['tag'],regime=t['regime'],role=r['role'],mode=r['mode'],candidate=r['candidate'],protocol_phase=r['phase'],truth=rec['truth'],wm=rec['predictions']['WM'],combined=rec['predictions']['error-only'],query_value=rec['answers']['error-only'] if rec['answers']['error-only'] is not None else -1,flags=flags.copy(),active=active,before_id=before_id,memory_id=after_id,response=response,overlaps=overlaps,changes=changes,progress=progress))
    before_id=after_id
assert at==len(inputs) and direct_taps==len(inputs)*32
assert events and len(edited_records)==32
control_hashes={
    '03':'5556397ff5f9a91bb62b2b4abdb05f033ac4c2ec490b7a16a95b793753e4129a',
    '04':'42650f94f507ca5a3bdcbe5f8d61aa1a531f10d94f9bc4ac2b941aa896369dc2'}
control_checks={}
for cell,arm in [('02','error-only'),('03','all-admit'),('04','S1AB-frozen')]:
    zp=B/f'NATIVE_{cell}.zip'
    if cell in control_hashes:assert sha(zp)==control_hashes[cell]
    zz=zipfile.ZipFile(zp); rr=load(B/f'native-{cell}/fixture/ROWS.json')
    oo=np.frombuffer(zz.read('RAW.outputs'),'<u4').reshape(len(rr),W)
    correct=0;count=0
    for ep,rec in enumerate(records[cell]):
        idx=[i for i,r in enumerate(rr)if r['episode']==ep and r['role']=='query'and r['net']==1]
        assert len(idx)==3
        counts=oo[idx,1:33].sum(axis=1).tolist()
        answer=counts.index(1)if counts.count(1)==1 and sum(counts)==1 else None
        assert answer==rec['answers'][arm]
        wm=np.asarray(rec['predictions']['WM'])
        combined=wm if answer is None else .25*wm+.75*np.eye(3)[answer]
        assert np.array_equal(combined,np.asarray(rec['predictions'][arm]))
        if audit[ep]['phase']==3 and audit[ep]['repeat']==1:
            count+=1;correct+=int(combined.argmax())==rec['truth']
    control_checks[arm]=dict(actual_native_query_answers_checked=144,new_second_encounter=dict(n=count,correct=correct),archive_sha256=sha(zp))
assert [control_checks[a]['new_second_encounter']['correct']for a in ['error-only','all-admit','S1AB-frozen']]==[16,1,1]
source=dict(native_archive=archive.name,native_sha256=sha(archive),source_forwards=n,physical=True,all_three_DEV_controls_native=True,media_summary_sha256=sha(M/'SUMMARY.json'),media_manifest_sha256=sha(M/'MANIFEST.jsonl'),scene_images='POSTHOC DETERMINISTIC SCENE RECONSTRUCTIONS; not camera or literal live-capture footage',query_reconstruction_equal_occurrences=288,selection='All144 native DEV episodes; all32 contexts, both gains. Every query and visible connection/threshold edit retained. No result-selected example.')
D=dict(schema=2,events=events,memory_versions=versions,catalog=catalog,images=images,geometry=dict(encoded_channels=P,content_neurons=32,total_NPUs=7,source_mapping=[1,3,0],memory_mapping=[3,2,0]),source=source,limits=['Recorded actual native responses and RAM edits; host-paced with static role clocks','Host encoder, labels and probability mixing; frozen GPU world-model weights','Images are source-verified posthoc scene reconstructions; native chip traces were captured during the experiment','Memory stores sparse descriptor/outcome associations, not the displayed images','Input highlights/overlaps derived from captured input bytes and pre-forward native RAM; 32 output taps directly measured','Error/match flags are measured source population aggregates, not individual source-neuron taps','Logical 3D geometry and editorial replay timing; no electrical-spike or physical die-layout claim','Selected repeated simulator contexts and designed admission policy; no fresh closed-loop control','Strong adaptive CPU controls also pass; no speed, energy or substrate-exclusion advantage','K32 CPU capacity diagnostic did not learn new corrections; N new-scene CPU gate failed with zero recall','Frozen-WM imagined frames retain the residual-decoder white smear; no repair or filtering'])
p=V/'data.json';p.write_text(json.dumps(D,separators=(',',':')))
receipt=dict(status='PASS',postprocessing_only=True,source_forwards=n,selected_visual_events=len(events),memory_versions=len(versions),direct_response_taps_checked=direct_taps,actual_connection_additions=total_edits,neurons_with_measured_connection_edits=len(edited_records),images_verified=sum(len(m['images'])for m in manifest),all_media_png_and_rgb_hashes_pass=True,all_event_taps_match_actual_input_and_prior_RAM=True,no_future_identity_annotation=True,native_control_output_checks=control_checks,data_sha256=sha(p),source=source)
(V/'DATA_QA.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
