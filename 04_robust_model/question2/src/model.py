"""Lightweight TFR adaptation, not a reproduction of published TFR-Net scores.

Official MIT-licensed Generator is reused verbatim. Text encoder, masked
attention/fusion, classification head and interval corruption are new adapters.
"""
from types import SimpleNamespace
from pathlib import Path
import os
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[1]/'pretrained/cache'))
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel, BertConfig
from tfr_generator import Generator


def masked_mean(x, mask):
    return (x*mask[...,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


class RobustModel(nn.Module):
    def __init__(self, cfg, pretrained=None, bert_config=None):
        super().__init__(); self.cfg=cfg
        if pretrained is not None:
            self.bert=BertModel.from_pretrained(str(pretrained),attn_implementation='eager')
        else:
            self.bert=BertModel(BertConfig.from_dict(bert_config))
        d=cfg.get('hidden',32); self.d=d
        dims=[self.bert.config.hidden_size,74,35]
        self.proj=nn.ModuleList([nn.Sequential(nn.Linear(x,d),nn.LayerNorm(d),nn.GELU()) for x in dims])
        self.pos=nn.Parameter(torch.randn(1,50,d)*.02)
        self.missing=nn.Parameter(torch.randn(3,d)*.02)
        self.null=nn.Parameter(torch.zeros(1,1,d))
        self.self_attn=nn.ModuleList([nn.TransformerEncoderLayer(d,4,4*d,.15,batch_first=True) for _ in range(3)])
        self.cross=nn.ModuleList([nn.MultiheadAttention(d,4,dropout=.1,batch_first=True) for _ in range(3)])
        self.cross_norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.gate=nn.Sequential(nn.Linear(3*d+1,32),nn.Tanh(),nn.Linear(32,1))
        self.head=nn.Sequential(nn.Linear(9*d+3,96),nn.GELU(),nn.Dropout(.25),nn.Linear(96,4))
        args=SimpleNamespace(dst_feature_dim_nheads=(d,4),feature_dims=dims,generatorModule='linear')
        self.reconstruct=nn.ModuleList([Generator(args,m) for m in ['text','audio','vision']])

    def forward(self,b,clean_targets=False):
        obs=b['obs'].clone(); content=b['content']; valid=b['valid']
        if self.cfg.get('text_only'):
            obs[:,:,1:]=False
        ids=b['ids'].clone()
        # Missing tokens are removed BEFORE contextual text encoding.
        ids[content & ~obs[:,:,0]]=103
        ids[~valid]=0
        text=self.bert(input_ids=ids,attention_mask=valid.long(),token_type_ids=b['segments']).last_hidden_state
        raw=[text,b['audio']*obs[:,:,1,None],b['vision']*obs[:,:,2,None]]
        hs=[]
        for m in range(3):
            h=self.proj[m](raw[m])*obs[:,:,m,None]
            if self.cfg.get('mask_aware',True): h=h+(content & ~obs[:,:,m])[...,None]*self.missing[m]
            hs.append(h+self.pos[:,:h.shape[1]])
        contexts=[]
        for m in range(3):
            own=self.self_attn[m](hs[m],src_key_padding_mask=~valid)
            others=[j for j in range(3) if j!=m]
            kv=torch.cat([hs[j] for j in others]+[self.null.expand(len(ids),-1,-1)],1)
            if self.cfg.get('mask_aware',True):
                unavailable=[~obs[:,:,j] for j in others]
            else: unavailable=[~valid for j in others]
            kpm=torch.cat(unavailable+[torch.zeros(len(ids),1,dtype=torch.bool,device=ids.device)],1)
            cross,_=self.cross[m](own,kv,kv,key_padding_mask=kpm,need_weights=False)
            contexts.append(torch.cat([hs[m],own,self.cross_norm[m](cross)],-1))
        pooled=torch.stack([masked_mean(h,content) for h in contexts],1)
        coverage=obs.float().sum(1)/content.sum(1,keepdim=True).clamp_min(1)
        if self.cfg.get('text_only'):
            weights=torch.zeros_like(coverage);weights[:,0]=1
        elif self.cfg.get('gating',True):
            logits=self.gate(torch.cat([pooled,coverage[...,None]],-1)).squeeze(-1)
            weights=(logits+torch.log(coverage+.05)).softmax(-1)
        else: weights=torch.ones_like(coverage)/3
        cov_in=coverage if self.cfg.get('mask_aware',True) else torch.zeros_like(coverage)
        pred=self.head(torch.cat([(pooled*weights[...,None]).flatten(1),cov_in],-1))
        result={'reg':3*torch.tanh(pred[:,0]),'logits':pred[:,1:],'weights':weights}
        if clean_targets:
            # Teacher targets are detached, in deterministic eval mode, and never
            # enter the prediction branch. Supervision only on synthetic deletions.
            was_training=self.bert.training
            self.bert.eval()
            with torch.no_grad():
                target_ids=b['ids'].clone()
                target_ids[content & ~b['original_obs'][:,:,0]]=103
                clean_text=self.bert(input_ids=target_ids,attention_mask=valid.long(),token_type_ids=b['segments']).last_hidden_state
            self.bert.train(was_training)
            targets=[clean_text,b['audio'],b['vision']]
            losses=[]
            for m in range(3):
                mask=b['removed'][:,:,m]
                reconstruction=self.reconstruct[m](contexts[m])
                err=F.smooth_l1_loss(reconstruction,targets[m].detach(),reduction='none').mean(-1)
                losses.append((err*mask).sum()/mask.sum().clamp_min(1))
            result['rec_loss']=sum(losses)/3
        return result
