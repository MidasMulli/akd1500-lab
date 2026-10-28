const {test}=require('node:test');
const assert=require('node:assert/strict');
const {makeEndpoint}=require('./endpoint.cjs');

test('default points at the local measured explorer',()=>{
  assert.equal(makeEndpoint().origin,'http://127.0.0.1:8892');
});
test('non-local addresses are rejected in the public desktop shell',()=>{
  assert.throws(()=>makeEndpoint('http://192.0.2.1:8892/'));
});
test('rejects arbitrary origins, credentials, protocols, ports and paths',()=>{
  for(const url of ['https://example.org/','file:///tmp/view.html','http://127.0.0.1:8893/','http://192.0.2.1.example.org:8892/','http://user:secret@127.0.0.1:8892/','http://127.0.0.1:8892/api/live','javascript:alert(1)'])
    assert.throws(()=>makeEndpoint(url),undefined,url);
});
test('drops unrelated startup parameters and never asks to start native work',()=>{
  const {target}=makeEndpoint('http://localhost:8892/?action=start&mode=native#run');
  assert.deepEqual([...target.searchParams],[['view','nodes'],['background','0'],['connections','1'],['desktop','1']]);
  assert.equal(target.hash,'');
});
