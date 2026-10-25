import unittest
import numpy as np
from admission import latency
from admission.latency_archive import check_run

class ConsensusLatencyTests(unittest.TestCase):
    def test_decode_handles_both_neither_and_each_answer(self):
        y = np.zeros((4,128), 'u1');y[0,88]=1;y[1,0]=1;y[2,[0,88]]=1
        self.assertEqual(latency.decode(y).tolist(), [0,1,-1,-1])

    def test_confirmation_is_read_only_and_source_silent(self):
        rows = latency.schedule('repeat')
        self.assertEqual(len(rows),3064)
        for i,e in enumerate(rows):
            if e['phase'].startswith('accepted_transfer') and e['role']=='transfer':
                probes=rows[i+1:i+1+latency.PROBES]
                self.assertEqual(len(probes),16)
                self.assertTrue(all(p['role']=='exam' and p['key']==e['key'] and p['labels']==[-1,-1,-1] for p in probes))

    def test_reliability_rejects_a_later_regression(self):
        rows=latency.schedule('gate');answer=np.array([e['actual'] for e in rows])
        t=np.zeros((len(rows),14),'i8');t[:,6]=10;t[:,9]=np.arange(len(rows))*20;t[:,10]=t[:,9]+10
        good=latency.convergence(rows,answer,t)
        self.assertEqual([c['teaching_attempts'] for c in good],[1,1])
        j=next(i for i,e in enumerate(rows) if e['phase']=='updated_exam_1' and e['key']==0)
        answer[j]^=1
        with self.assertRaises(AssertionError):latency.convergence(rows,answer,t)

    def test_critical_path_excludes_audit_and_confirmation_queries(self):
        rows=latency.schedule('gate');answer=np.array([e['actual'] for e in rows])
        t=np.zeros((len(rows),14),'i8');t[:,6]=10;t[:,7:9]=1000;t[:,9]=np.arange(len(rows))*3000;t[:,10]=t[:,9]+10
        a,b=latency.convergence(rows,answer,t)
        self.assertEqual(a['learning_path_ns'],20)
        self.assertEqual(b['learning_path_ns'],40)
        self.assertEqual(a['learner_learning_path_ns'],20)
        self.assertEqual(b['learner_learning_path_ns'],20)
        self.assertEqual(a['confirming_queries_ns'],160)
        self.assertGreater(a['observed_until_confirmed_ns'],a['operational_until_confirmed_ns'])

    def test_audit_rejects_unaccounted_time_and_wrong_output(self):
        rows=latency.schedule('gate');n=len(rows)
        t=np.zeros((n,14),'i8');t[:,6]=10;t[:,9]=np.arange(n)*20;t[:,10]=t[:,9]+10
        answer=np.array([e['actual'] for e in rows],'i4')
        ref=dict(events=rows,cpu=np.column_stack([answer,answer]))
        arrays=dict(timing=t.ravel(),answers=answer.copy())
        report=dict(agreed_update_batch_ns=100,agreed_update_batch_operations=10,device_opened=False)
        run=dict(mode='dictionary')
        self.assertEqual(len(check_run(run,report,arrays.__getitem__,ref)),2)
        bad=t.copy();bad[10,6]+=1;arrays['timing']=bad.ravel()
        with self.assertRaises(AssertionError):check_run(run,report,arrays.__getitem__,ref)
        arrays['timing']=t.ravel();arrays['answers'][0]^=1
        with self.assertRaises(AssertionError):check_run(run,report,arrays.__getitem__,ref)

if __name__=='__main__':unittest.main()
