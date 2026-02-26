
import random
import math 
import torch 
import torch.nn.functional as F
from einops import rearrange
from flood.utils.benchmark import benchmark_func, output_check

from flood.ops.seg_la import seg_la_fwd


def chunk_simple_gla_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    initial_state = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    scale = None,
):
    q, k, v = map(lambda x: rearrange(x, 'b t h ... -> b h t ...'), [q, k, v])
    if g is not None:
        g = rearrange(g, 'b t h ... -> b h t ...')
    if scale is None:
        scale = 1.0 / q.shape[-1] ** 0.5

    T = q.shape[-2]
    BT = chunk_size
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, g = map(lambda x: x.to(torch.float32), [q, k, v, g])
    decay = -g
    b, h, t, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    q, k, v, decay = map(lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=chunk_size), [q, k, v, decay.unsqueeze(-1)])
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state.float()
    o = torch.zeros_like(v)
    for i in range(0, t // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i])
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_i
        S = S * decay[:, :, i, -1, None, None].exp() + \
            (k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_i
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, 'b h n c d -> b h (n c) d')
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


class Meta:
    def __init__(self) -> None:
        pass


def _torch_linear_attn(q, k, v, s, s_scales, decay_scales):
    dtype = q.dtype
    q = q.float()
    k = k.float()
    v = v.float()
    s = s.float()
    bs, q_len, q_heads, head_dim = q.shape
    k_heads = k.shape[2]
    k_len = k.shape[1]
    assert q_len == k_len
    softmax_scale = 1.0/math.sqrt(head_dim)
    query = q.transpose(1, 2) * softmax_scale  # [bs, head, len, dim]
    key = torch.permute(k, (0, 2, 3, 1))  # [bs, head, dim, len]
    value = v.transpose(1, 2)  # [bs, head, len, dim]
    if k_heads != q_heads:
        g = q_heads // k_heads
        key = torch.repeat_interleave(key, g, dim=1)
        value = torch.repeat_interleave(value, g, dim=1)

    arr = torch.arange(q_len, dtype=torch.float32, device=q.device)
    decay_matrix = arr.view(-1,1) - arr.view(1,-1)
    decay_matrix = torch.exp(-decay_scales[:,None,None] * decay_matrix[None])
    decay_matrix = torch.tril(decay_matrix, 0)

    score = torch.matmul(query, key)
    score *= decay_matrix[None]
    att = torch.matmul(score, value)

    decay_arr = torch.exp(-decay_scales[:,None,None]*(arr[:,None]+1))
    att += torch.matmul(query*decay_arr, s*s_scales[:,None,None,None]) 

    att = torch.reshape(att.transpose(1, 2),
                        [bs, q_len, q_heads, head_dim]).contiguous()

    decay_key = key*torch.exp(-decay_scales[:,None,None]*(q_len-1-arr))
    state = decay_key@value + s*torch.exp(-decay_scales[:,None,None])

    return att.to(dtype), state

def torch_linear_attn(q, k, v, s, s_scales, decay_scales, mask=None):
    if mask is None:
        return _torch_linear_attn(q, k, v, s, s_scales, decay_scales)

    outputs = []
    for i in range(mask.size(-1)):
        m = mask[0,i,:]
        indices = torch.where(m>0)[0]
        q_ = q[:, indices]
        k_ = k[:, indices]
        v_ = v[:, indices]
        o_, _ = _torch_linear_attn(q_, k_, v_, s, s_scales, decay_scales)
        outputs.append(o_[:,-1:])
    outputs = torch.cat(outputs, 1)
    return outputs, s


def get_seg_attn_meta(qls, kls, s_scales, mask=None):
    device = 'cuda:0'
    bs = len(qls)
    q_offsets = [0]  # [bs+1]
    k_offsets = [0]  # [bs+1]
    q_lengths = []  # [bs]
    k_lengths = []  # [bs]
    max_q_length = max(qls)
    max_k_length = max(kls)
    for i, ql in enumerate(qls):
        kl = kls[i]
        q_offsets.append(q_offsets[-1] + ql)
        q_lengths.append(ql)
        k_offsets.append(k_offsets[-1] + kl)
        k_lengths.append(kl)

    q_offsets = torch.tensor(q_offsets, device=device, dtype=torch.int32)
    q_lengths = torch.tensor(q_lengths, device=device, dtype=torch.int32)
    s_offsets = torch.arange(bs, device=device, dtype=torch.int32)

    meta = Meta()
    meta.batch_size = bs
    meta.q_offsets = q_offsets
    meta.q_lengths = q_lengths
    meta.s_offsets = s_offsets
    meta.max_q_length = max_q_length
    meta.max_k_length = max_k_length
    meta.mask = mask
    meta.qls = qls
    meta.kls = kls
    meta.s_scales = s_scales
    return meta



def make_input(qo_heads=16, kv_heads=16, dim=128, qls=[1024,1024], kls=[1024,1024]):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    qs = []
    ks = []
    vs = []
    for i, ql in enumerate(qls):
        kvl = kls[i]
        q = torch.randn(ql, qo_heads, dim, dtype=dtype, device=device)
        k = torch.randn(kvl, kv_heads, dim, dtype=dtype, device=device)
        v = torch.randn(kvl, kv_heads, dim, dtype=dtype, device=device)
        qs.append(q)
        ks.append(k)
        vs.append(v)

    q = torch.cat(qs, 0)
    k = torch.cat(ks, 0)
    v = torch.cat(vs, 0)

    return q, k, v


def test_prefill_seg_attn(bs=2, qo_heads=16, kv_heads=16, dim=128, qls=[1024,1024], digest=False, bench=False):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    kls = qls
    
    assert all([x<=kls[i] for i,x in enumerate(qls)])

    ref_bytes = sum(qls) * qo_heads * dim * 8 + bs * qo_heads * dim * dim * 8

    q, k, v = make_input(qo_heads=qo_heads, kv_heads=kv_heads, dim=dim, qls=qls, kls=kls)

    s = torch.randn(bs, kv_heads, dim, dim, dtype=torch.float32, device=device)
    s_scales = torch.ones((bs,), device=device, dtype=dtype)

    decay_scales = 2**(-0.5 * torch.arange(1, qo_heads+1, dtype=torch.float32, device=device))
    # decay_scales = 2**(-0.5 * torch.ones(qo_head, dtype=torch.float32, device=device))

    seg_attn_meta = get_seg_attn_meta(qls, kls, s_scales)

    org_output, state_ref = chunk_simple_gla_ref(q.view(bs, qls[0], qo_heads, dim), 
                                    k.view(bs, kls[0], kv_heads, dim), 
                                    v.view(bs, kls[0], kv_heads, dim),
                                    decay_scales.view(1,1,-1).expand(bs, qls[0], qo_heads),
                                    initial_state=s.clone().detach(),
                                    output_final_state=True)
    org_output = torch.reshape(org_output, (bs*qls[0], qo_heads, dim)).to(dtype)
    opt_output = seg_la_fwd(q, k, v, s, decay_scales, seg_attn_meta)

    output_check(org_output, opt_output, name='output', rtol=0.1, atol=0.1)
    output_check(state_ref, s, name='state', rtol=0.01, atol=0.01)

    if digest:
        print(
            f"org max:{torch.max(org_output).item():.3f} min:{torch.min(org_output).item():.3f}")
        print(
            f"opt max:{torch.max(opt_output).item():.3f} min:{torch.min(opt_output).item():.3f}")

        print("org_output[:,0,0]", org_output[:, 0, 0])
        print("opt_output[:,0,0]", opt_output[:, 0, 0])

        print("org_output[0,:,0]", org_output[0, :, 0])
        print("opt_output[0,:,0]", opt_output[0, :, 0])

        print("org_output[0,0,:]", org_output[0, 0, :])
        print("opt_output[0,0,:]", opt_output[0, 0, :])

    if bench:
        benchmark_func(seg_la_fwd, q, k, v, s, decay_scales, seg_attn_meta, ref_bytes=ref_bytes)


def test_decode_seg_attn(bs=2, qo_heads=16, kv_heads=16, dim=128, digest=False, bench=False):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16
    
    qls = [1]*bs
    kls = [1]*bs

    ref_bytes = sum(qls) * qo_heads * dim * 8 + bs * qo_heads * dim * dim * 8

    q, k, v = make_input(qo_heads=qo_heads, kv_heads=kv_heads, dim=dim, qls=qls, kls=qls)

    s = torch.randn(bs, kv_heads, dim, dim, dtype=torch.float32, device=device)
    s_scales = torch.ones((bs,), device=device, dtype=dtype)

    decay_scales = 2**(-0.5 * torch.arange(1, qo_heads+1, dtype=torch.float32, device=device))
    # decay_scales = 2**(-0.5 * torch.ones(qo_head, dtype=torch.float32, device=device))

    seg_attn_meta = get_seg_attn_meta(qls, kls, s_scales)

    org_output, state_ref = chunk_simple_gla_ref(q.view(bs, qls[0], qo_heads, dim), 
                                    k.view(bs, kls[0], kv_heads, dim), 
                                    v.view(bs, kls[0], kv_heads, dim),
                                    decay_scales.view(1,1,-1).expand(bs, qls[0], qo_heads),
                                    initial_state=s.clone().detach(),
                                    output_final_state=True)
    org_output = torch.reshape(org_output, (bs*qls[0], qo_heads, dim)).to(dtype)
    opt_output = seg_la_fwd(q, k, v, s, decay_scales, seg_attn_meta)

    output_check(org_output, opt_output, name='output', rtol=0.1, atol=0.1)
    output_check(state_ref, s, name='state', rtol=0.01, atol=0.01)

    if digest:
        print(
            f"org max:{torch.max(org_output).item():.3f} min:{torch.min(org_output).item():.3f}")
        print(
            f"opt max:{torch.max(opt_output).item():.3f} min:{torch.min(opt_output).item():.3f}")

        print("org_output[:,0,0]", org_output[:, 0, 0])
        print("opt_output[:,0,0]", opt_output[:, 0, 0])

        print("org_output[0,:,0]", org_output[0, :, 0])
        print("opt_output[0,:,0]", opt_output[0, :, 0])

        print("org_output[0,0,:]", org_output[0, 0, :])
        print("opt_output[0,0,:]", opt_output[0, 0, :])

    if bench:
        benchmark_func(seg_la_fwd, q, k, v, s, decay_scales, seg_attn_meta, ref_bytes=ref_bytes)


def test_mtp_seg_attn(bs=2, qo_heads=16, kv_heads=16, dim=128, mask_size=4, digest=False, bench=False):
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    qls = [mask_size] * bs
    kls = qls

    assert all([x<=kls[i] for i,x in enumerate(qls)])

    ref_bytes = sum(qls) * qo_heads * dim * 8 + bs * mask_size * qo_heads * dim * dim * 8 

    q, k, v = make_input(qo_heads=qo_heads, kv_heads=kv_heads, dim=dim, qls=qls, kls=kls)

    s = torch.randn(bs, kv_heads, dim, dim, dtype=torch.float32, device=device)
    s_scales = torch.ones((bs,), device=device, dtype=dtype)

    decay_scales = 2**(-0.5 * torch.arange(1, qo_heads+1, dtype=torch.float32, device=device))
    # decay_scales = 2**(-0.5 * torch.ones(qo_head, dtype=torch.float32, device=device))

    seg_attn_meta = get_seg_attn_meta(qls, kls, s_scales)

    states_ref = []
    org_outputs = []
    state_ref = s
    for i in range(mask_size):
        org_output, state_ref = torch_linear_attn(q.view(bs, qls[0], qo_heads, dim)[:,i:i+1], 
                                        k.view(bs, kls[0], kv_heads, dim)[:,i:i+1], 
                                        v.view(bs, kls[0], kv_heads, dim)[:,i:i+1],
                                        state_ref,
                                        s_scales,
                                        decay_scales)
        states_ref.append(state_ref)
        org_outputs.append(org_output)
    state_ref = torch.stack(states_ref, 1).view(bs, mask_size, kv_heads, dim, dim)
    org_output = torch.cat(org_outputs, 1).view(bs*qls[0], qo_heads, dim)
    caches = torch.zeros(bs, mask_size, kv_heads, dim, dim, dtype=torch.float32, device=device)
    opt_output = seg_la_fwd(q, k, v, s, decay_scales, seg_attn_meta, caches=caches)

    output_check(org_output, opt_output, name='output', rtol=0.1, atol=0.1)
    output_check(state_ref, caches, name='state', rtol=0.01, atol=0.01)

    if digest:
        print(
            f"org max:{torch.max(org_output).item():.3f} min:{torch.min(org_output).item():.3f}")
        print(
            f"opt max:{torch.max(opt_output).item():.3f} min:{torch.min(opt_output).item():.3f}")

        print("org_output[:,0,0]", org_output[:, 0, 0])
        print("opt_output[:,0,0]", opt_output[:, 0, 0])

        print("org_output[0,:,0]", org_output[0, :, 0])
        print("opt_output[0,:,0]", opt_output[0, :, 0])

        print("org_output[0,0,:]", org_output[0, 0, :])
        print("opt_output[0,0,:]", opt_output[0, 0, :])

    if bench:
        benchmark_func(seg_la_fwd, q, k, v, s, decay_scales, seg_attn_meta, caches=caches, ref_bytes=ref_bytes)


def test_speculative_seg_attn(bs=1, qo_heads=16, kv_heads=16, dim=128,  mask_size=16, dump=False, digest=False, bench=False):
    # assert bs==1
    device = torch.device('cuda:0')
    dtype = torch.bfloat16

    qls = [mask_size] * bs
    kls = [mask_size] * bs

    assert all([x==mask_size for x in qls])

    masks = torch.ones((bs, mask_size, mask_size), 
                            dtype=torch.int8,
                            device=device)
    masks[:, 5:, 1:5] = 0
    masks[:, 8:, 5:8] = 0
    masks = torch.tril(masks, 0)

    ref_bytes = sum(qls) * qo_heads * dim * 12 + bs * qo_heads * dim * dim * 4

    q, k, v = make_input(qo_heads=qo_heads, kv_heads=kv_heads, dim=dim, qls=qls, kls=kls)

    s = torch.randn(bs, kv_heads, dim, dim, dtype=torch.float32, device=device)
    s_scales = torch.ones((bs,), device=device, dtype=dtype)
    if dump:
        key_cache = torch.zeros(bs, mask_size, kv_heads, dim, dtype=dtype, device=device)
        value_cache = torch.zeros(bs, mask_size, kv_heads, dim, dtype=dtype, device=device)
    else:
        key_cache = None 
        value_cache = None

    decay_scales = 2**(-0.5 * torch.arange(1, qo_heads+1, dtype=torch.float32, device=device))
    # decay_scales = 2**(-0.5 * torch.ones(qo_head, dtype=torch.float32, device=device))

    seg_attn_meta = get_seg_attn_meta(qls, kls, s_scales, mask=masks)

    org_output, state_ref = torch_linear_attn(q.view(bs, qls[0], qo_heads, dim), 
                                    k.view(bs, kls[0], kv_heads, dim), 
                                    v.view(bs, kls[0], kv_heads, dim),
                                    s,
                                    s_scales,
                                    decay_scales,
                                    mask=seg_attn_meta.mask) 
    org_output = torch.reshape(org_output, (bs*qls[0], qo_heads, dim)).to(dtype)
    opt_output = seg_la_fwd(q, k, v, s, decay_scales, seg_attn_meta, key_cache=key_cache, value_cache=value_cache)

    output_check(org_output, opt_output, name='output', rtol=0.1, atol=0.1)
    output_check(state_ref, s, name='state', rtol=0.01, atol=0.01)  # not updated
    if dump:
        output_check(k.view(bs, kls[0], kv_heads, dim), key_cache, name='key_cache', rtol=0.01, atol=0.01)
        output_check(v.view(bs, kls[0], kv_heads, dim), value_cache, name='value_cache', rtol=0.01, atol=0.01)

    if digest:
        print(
            f"org max:{torch.max(org_output).item():.3f} min:{torch.min(org_output).item():.3f}")
        print(
            f"opt max:{torch.max(opt_output).item():.3f} min:{torch.min(opt_output).item():.3f}")

        print("org_output[:,0,0]", org_output[:, 0, 0])
        print("opt_output[:,0,0]", opt_output[:, 0, 0])

        print("org_output[0,:,0]", org_output[0, :, 0])
        print("opt_output[0,:,0]", opt_output[0, :, 0])

        print("org_output[0,0,:]", org_output[0, 0, :])
        print("opt_output[0,0,:]", opt_output[0, 0, :])

    if bench:
        benchmark_func(seg_la_fwd, q, k, v, s, decay_scales, seg_attn_meta, 
                       key_cache=key_cache, value_cache=value_cache, ref_bytes=ref_bytes)

if __name__ == '__main__':
    test_prefill_seg_attn(bs=2, qo_heads=16, kv_heads=16, dim=128, qls=[1024,1024], digest=False, bench=True)
    test_decode_seg_attn(bs=156, qo_heads=16, kv_heads=16, dim=128, digest=False, bench=True)
    test_mtp_seg_attn(bs=64, qo_heads=16, kv_heads=16, dim=128, mask_size=4, digest=False, bench=True)
    test_speculative_seg_attn(bs=32, qo_heads=16, kv_heads=16, dim=128, mask_size=16, dump=False, digest=False, bench=True)
    test_speculative_seg_attn(bs=32, qo_heads=16, kv_heads=16, dim=128, mask_size=16, dump=True, digest=False, bench=True)
