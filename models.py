import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel
from timm import create_model
from transformers import GPT2TokenizerFast
from types import SimpleNamespace


tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

class GPT2Attention(nn.Module):
    def __init__(self,config:SimpleNamespace):
        super(GPT2Attention,self).__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, "embedding dim must be divisible by num heads"
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        self.c_attn = nn.Linear(self.embed_dim,self.embed_dim*3)
        self.scale = self.head_size ** -0.5
        
        self.register_buffer('mask',torch.tril(torch.ones(1,1,self.seq_len,self.seq_len)))
        self.c_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        
    def forward(self,x:torch.Tensor)-> torch.Tensor:
        b,t,c = x.shape 
        
        q,k,v = self.c_attn(x).chunk(3,dim=-1)
        q = q.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        k = k.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        
        qk_t = (q@k.transpose(-2,-1))*self.scale
        qk_t = qk_t.masked_fill(self.mask[:,:,:t,:t]==0,float('-inf'))
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        attention = weights@v
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c)
        
        out = self.c_proj(attention)
        return self.resid_dropout(out)

class GPT2CrossAttention(nn.Module):
    def __init__(self,config:SimpleNamespace):
        super(GPT2CrossAttention,self).__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim %self.n_heads == 0, "embedding dim must be divisible by num heads"
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len
        
        self.q = nn.Linear(self.embed_dim,self.embed_dim)
        self.k = nn.Linear(self.embed_dim,self.embed_dim)
        self.v = nn.Linear(self.embed_dim,self.embed_dim)
        self.scale = self.head_size ** -0.5
        
        self.c_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        self.apply(self._init_weights)
    
    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            nn.init.normal_(module.weight,mean=0.0,std=0.02)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self,q:torch.Tensor,k:torch.Tensor,v:torch.Tensor)->torch.Tensor:
        b,t,c = q.shape
        
        q,k,v = self.q(q),self.k(k),self.v(v)
        
        q = q.view(b,q.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        k = k.view(b,k.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,v.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        
        qk_t = (q@k.transpose(-2,-1))*self.scale
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)
        
        attention = weights@v
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c)
        
        out = self.c_proj(attention)
        return self.resid_dropout(out)
    
class GPT2MLP(nn.Module):
    def __init__(self,config:SimpleNamespace):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout
        self.c_fc = nn.Linear(self.embed_dim,self.embed_dim*self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim*self.mlp_ratio,self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return self.dropout(x)
    
class GPT2Block(nn.Module):
    def __init__(self,config:SimpleNamespace):
        super(GPT2Block,self).__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)
    
    def forward(self,x:torch.Tensor,enc_out:torch.Tensor)->torch.Tensor:  
        x = x+self.attn(self.ln_1(x))
        x = x+self.cross_attn(self.ln_2(x),enc_out,enc_out)
        x = x+self.mlp(self.ln_3(x))
        return x
    
class VisionGPT2Model(nn.Module):
    def __init__(self,config:SimpleNamespace):
        super(VisionGPT2Model,self).__init__()
        self.config = config
        vit = create_model('vit_base_patch16_224',pretrained=True,num_classes=0)
        self.patch_embed = vit.patch_embed
        num_patches = self.patch_embed.num_patches
        self.cls_token = vit.cls_token
        embed_len = num_patches + vit.num_prefix_tokens
        self.pos_embed = vit.pos_embed
        self.blocks = nn.ModuleList([vit.blocks[i] for i in range(config.depth)])
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size,config.embed_dim),
            wpe = nn.Embedding(config.seq_len,config.embed_dim),
            drop = nn.Dropout(config.emb_dropout),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f = nn.LayerNorm(config.embed_dim),
        ))
        self.lm_head = nn.Linear(config.embed_dim,config.vocab_size,bias= False)
        self.transformer.wte.weight = self.lm_head.weight
        
    def _pos_embed(self,x:torch.Tensor)->torch.Tensor:
        pos_embed = self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0],-1,-1),x),dim =1)
        x = x+pos_embed
        return x
    
    def pretrained_layers_trainable(self,t:bool = False)->None:
        layers =[
            self.cls_token,self.patch_embed,self.pos_embed,self.blocks,
            self.transformer.wte,self.transformer.wpe,
            self.transformer.ln_f,self.lm_head
        ]
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ]for i in range(self.config.depth)]
        
        for l in gpt_layers:
            layers.extend(l)
        
        for layer in layers:
            if not isinstance(layer,nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = t
            else:
                layer.requires_grad = t
                
        total_frozen_params = sum([p.numel() for p in self.parameters() if not p.requires_grad])
        print(f"{total_frozen_params =}")
        
    def unfreeze_gpt_layers(self)->None:
        gpt_layers = [[
            self.transformer.h[i].ln_1,self.transformer.h[i].ln_2,
            self.transformer.h[i].attn,self.transformer.h[i].mlp
        ]for i in range(self.config.depth)]
        
        flatten = []
        
        for l in gpt_layers:
            flatten.extend(l)
        
        for layer in flatten:
            if not isinstance(layer,nn.Parameter):
                for p in layer.parameters():
                    p.requires_grad = True
            else:
                layer.requires_grad = True
                
    @classmethod
    def from_pretrained(self,config:SimpleNamespace):
        model = VisionGPT2Model(config)
        sd = model.state_dict()
        keys = sd.keys()
        ignore_matches = ['blocks.','cross_attn.','ln_3','cls_token',
                         'pos_embed','patch_embed.','.attn.mask']
        vit_keys = [key for key in keys if any(match in key for match in ignore_matches)]
        gpt_keys = [key for key in keys if key not in vit_keys]
        gpt2_small = GPT2LMHeadModel.from_pretrained('gpt2')
        sd_hf = gpt2_small.state_dict()
        hf_keys = sd_hf.keys()
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.masked_bias')]
        hf_keys = [k for k in hf_keys if not k.endswith('.attn.bias')]
        transposed = ['attn.c_attn.weight','attn.c_proj.weight',
                     'mlp.c_fc.weight','mlp.c_proj.weight']
        
        for k in hf_keys:
            if any(match in k for match in ignore_matches):
                continue
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
                    
        model.load_state_dict(sd)
        return model

    def forward(self,image:torch.Tensor,input_ids:torch.Tensor,labels:None|torch.Tensor=None)->torch.Tensor:
        image = self.patch_embed(image)
        image = self._pos_embed(image)
        token_embeddings = self.transformer.wte(input_ids)
        pos_embs = torch.arange(0,input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        input_ids = self.transformer.drop(token_embeddings+positional_embeddings)
        
        for i in range(self.config.depth):
            image = self.blocks[i](image)
            input_ids = self.transformer.h[i](input_ids,image)
        input_ids = self.transformer.ln_f(input_ids)
        
        if labels is not None:
            lm_logits = self.lm_head(input_ids)
            loss = F.cross_entropy(lm_logits.view(-1,lm_logits.shape[-1]),labels.view(-1))
            return loss
        lm_logits = self.lm_head(input_ids[:,[-1],:])
        return lm_logits
    
    def generate(self,image:torch.Tensor,
                 sequence:torch.Tensor,
                 max_tokens:int =50,
                 temp:float =1.0,
                 deter:bool =False) -> torch.Tensor:
        
        for _ in range(max_tokens):
            out = self(image,sequence)
            out = out[:,-1,:]/temp
            probs = F.softmax(out,dim=-1)
            if deter:
                next_token = torch.argmax(probs,dim=-1,keepdim=True)
            else:
                next_token = torch.multinomial(probs,num_samples=1)
            
            sequence = torch.cat([sequence,next_token],dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        return sequence.cpu().flatten()