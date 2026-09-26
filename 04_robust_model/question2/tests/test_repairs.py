"""Regression tests for independently audited defects."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from data import corruption,pattern_eligible,deletion_count
from protocol import VERSION,completed_run_is_valid,sha256
from train import final_metric,selection_score

class Repairs(unittest.TestCase):
    def test_masks_pair_across_modalities_subset_and_order(self):
        c=np.zeros((35,50),bool)
        for i in range(35):c[i,1:15+i]=True
        o=np.repeat(c[:,:,None],3,axis=2);ids=np.array([f'sample-{i}' for i in range(35)])
        ref=corruption(c,o,seed=42,sample_ids=ids,rate=.3,modalities=[0])[0]
        for mods in ([0,1],[0,2],[0,1,2],[2,1,0]):
            got=corruption(c,o,seed=42,sample_ids=ids,rate=.3,modalities=mods)[0]
            np.testing.assert_array_equal(ref[:,:,0],got[:,:,0])
        ix=np.array([30,4,12,2,0])
        got=corruption(c[ix],o[ix],seed=42,sample_ids=ids[ix],rate=.3,modalities=[0,1,2])[0]
        np.testing.assert_array_equal(ref[ix,:,0],got[:,:,0])

    def test_strict_two_blocks_and_short_population(self):
        c=np.zeros((49,50),bool)
        for i in range(49):c[i,:i+1]=True
        o=np.repeat(c[:,:,None],3,axis=2);ids=np.arange(49)
        for rate in (.1,.3,.5,.7,1.):
            eligible=pattern_eligible(c,rate)
            for seed in range(20):
                _,removed=corruption(c[eligible],o[eligible],seed=seed,sample_ids=ids[eligible],rate=rate,pattern='two_blocks')
                for row,n in zip(removed,c[eligible].sum(1)):
                    for m in range(3):
                        ix=np.flatnonzero(row[:,m]);self.assertEqual(len(ix),deletion_count(int(n),rate))
                        self.assertEqual(1+int((np.diff(ix)>1).sum()),2)
            with self.assertRaises(ValueError):
                corruption(c,o,seed=1,sample_ids=ids,rate=rate,pattern='two_blocks')

    def test_selection_uses_delivered_projection(self):
        y=np.array([0.,-1.,1.]);cls=np.array([1,0,2]);raw=np.array([2.,-1.,1.]);logits=np.eye(3)[cls]
        m=final_metric(y,cls,raw,logits)
        self.assertEqual(m['mae'],0.);self.assertGreater(m['raw_mae'],0.)
        self.assertEqual(selection_score({'clean':m,'T30':m}),0.)

    def test_completion_requires_matching_usable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);cfg={'protocol':VERSION,'variant':'x'}
            (p/'config.json').write_text(json.dumps(cfg));(p/'finished.json').write_text('{}')
            self.assertFalse(completed_run_is_valid(p,cfg))
            torch.save({'config':cfg,'model':{'w':torch.zeros(1)},'score':.5},p/'best.pt')
            (p/'finished.json').write_text(json.dumps({'checkpoint_sha256':sha256(p/'best.pt'),'best_score':.5}))
            self.assertTrue(completed_run_is_valid(p,cfg))
            self.assertFalse(completed_run_is_valid(p,{**cfg,'variant':'changed'}))
            with (p/'best.pt').open('ab') as f:f.write(b'corruption')
            self.assertFalse(completed_run_is_valid(p,cfg))

if __name__=='__main__':unittest.main()
