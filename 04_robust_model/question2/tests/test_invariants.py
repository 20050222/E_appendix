"""Meaningful regression guards: corruption placement and missing-token leakage."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from data import corruption,adapt
from model import RobustModel
from train import batch,project_intensity


class Invariants(unittest.TestCase):
    def test_output_polarity_consistency(self):
        p=np.array([.9,-.2,-.7,.4]);logits=np.array([[5,0,0],[0,5,0],[0,0,5],[0,0,5]])
        q=project_intensity(p,logits)
        self.assertLess(q[0],0);self.assertEqual(q[1],0);self.assertGreater(q[2],0);self.assertAlmostEqual(q[3],.4)

    def test_block_only_in_content_and_deterministic(self):
        content=np.zeros((4,50),bool);content[:,1:21]=1
        obs=np.repeat(content[:,:,None],3,axis=2)
        a,r=corruption(content,obs,np.random.default_rng(7),rate=.3,modalities=[1])
        b,_=corruption(content,obs,np.random.default_rng(7),rate=.3,modalities=[1])
        self.assertTrue(np.array_equal(a,b));self.assertTrue(np.all(a<=obs))
        self.assertEqual(r[:,:,0].sum()+r[:,:,2].sum(),0)
        for row in r[:,:,1]:
            ix=np.flatnonzero(row);self.assertEqual(len(ix),6);self.assertTrue(np.all(np.diff(ix)==1))

    def test_missing_tokens_and_features_do_not_leak(self):
        torch.set_num_threads(2);torch.manual_seed(5)
        tb=np.zeros((2,3,50),np.int64);tb[:,0,:7]=[101,2000,2001,2002,2003,2004,102];tb[:,1,:7]=1
        s=dict(text_bert=tb,audio=np.ones((2,50,74),np.float32),vision=np.ones((2,50,35),np.float32))
        s['audio'][:,7:]=0;s['vision'][:,7:]=0
        d=adapt(s);kept=d['obs'].copy();kept[:,2:5,:]=False
        model=RobustModel(dict(hidden=16,gating=True,mask_aware=True),bert_config=dict(vocab_size=30522,hidden_size=32,num_hidden_layers=1,num_attention_heads=2,intermediate_size=64,max_position_embeddings=512)).eval()
        b=batch(d,np.arange(2),torch.device('cpu'),kept)
        with torch.no_grad():p=model(b)
        b['ids'][:,2:5]=9999;b['audio'][:,2:5]=999;b['vision'][:,2:5]=-999
        with torch.no_grad():q=model(b)
        torch.testing.assert_close(p['reg'],q['reg'],atol=0,rtol=0)
        torch.testing.assert_close(p['logits'],q['logits'],atol=0,rtol=0)
        b['obs'][:]=False
        with torch.no_grad():r=model(b)
        self.assertTrue(torch.isfinite(r['reg']).all())
        self.assertTrue(torch.isfinite(r['logits']).all())
        empty=adapt(dict(text_bert=np.zeros((1,3,50),np.int64),audio=np.zeros((1,50,74),np.float32),vision=np.zeros((1,50,35),np.float32)))
        with torch.no_grad():r=model(batch(empty,np.array([0]),torch.device('cpu')))
        self.assertTrue(torch.isfinite(r['reg']).all())
        self.assertTrue(torch.isfinite(r['logits']).all())

    def test_padding_and_special_tokens_are_not_missing_content(self):
        tb=np.zeros((1,3,50),np.int64);tb[0,0,:5]=[101,2000,0,2001,102];tb[0,1,:5]=[1,1,0,1,1]
        d=adapt(dict(text_bert=tb,audio=np.zeros((1,50,74),np.float32),vision=np.zeros((1,50,35),np.float32)))
        self.assertEqual(d['valid'].sum(),5)
        self.assertEqual(d['content'].sum(),3)
        self.assertTrue(d['content'][0,2]);self.assertFalse(d['obs'][0,2,0])
        self.assertFalse(d['content'][0,0]);self.assertFalse(d['content'][0,4]);self.assertFalse(d['content'][0,5])

if __name__=='__main__':unittest.main()
