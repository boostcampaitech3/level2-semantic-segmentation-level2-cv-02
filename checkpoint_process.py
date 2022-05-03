import numpy as np
import torch
from scipy import interpolate

def put_checkpoint(model, state_dict, cfg):
    window_size = (cfg.model.backbone.img_size[0] // cfg.model.backbone.patch_size, 
                   cfg.model.backbone.img_size[1] // cfg.model.backbone.patch_size)
    
    except_keys = ['decode_head.conv_seg.weight', 
                   'decode_head.conv_seg.bias', 
                   'auxiliary_head.conv_seg.weight', 
                   'auxiliary_head.conv_seg.bias']
    all_keys = list(state_dict.keys())
    for key in all_keys:
        if ('relative_position_index' in key) or (key in except_keys):
            state_dict.pop(key)
        
        if 'relative_position_bias_table' in key:
            rel_pos_bias = state_dict[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            dst_patch_shape = window_size
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens)**0.5)
            dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
            if src_size != dst_size:
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]
                new_rel_pos_bias = geometric_sequence_interpolation(src_size, dst_size, rel_pos_bias, num_attn_heads)
                new_rel_pos_bias = torch.cat((new_rel_pos_bias, extra_tokens), dim=0)
                state_dict[key] = new_rel_pos_bias
                
    return state_dict


def geometric_sequence_interpolation(src_size, dst_size, sequence, num):

        def geometric_progression(a, r, n):
            return a * (1.0 - r**n) / (1.0 - r)

        left, right = 1.01, 1.5
        while right - left > 1e-6:
            q = (left + right) / 2.0
            gp = geometric_progression(1, q, src_size // 2)
            if gp > dst_size // 2:
                right = q
            else:
                left = q
        
        dis = []
        cur = 1
        for i in range(src_size // 2):
            dis.append(cur)
            cur += q**(i + 1)
        r_ids = [-_ for _ in reversed(dis)]
        x = r_ids + [0] + dis
        y = r_ids + [0] + dis
        t = dst_size // 2.0
        dx = np.arange(-t, t + 0.1, 1.0)
        dy = np.arange(-t, t + 0.1, 1.0)
        
        new_sequence = []
        for i in range(num):
            z = sequence[:, i].view(src_size, src_size).float().numpy()
            f = interpolate.interp2d(x, y, z, kind='cubic')
            new_sequence.append(
                torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(sequence))
        new_sequence = torch.cat(new_sequence, dim=-1)
        
        return new_sequence