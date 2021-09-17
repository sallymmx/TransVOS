import torch
import torch.nn.functional as F
from torch import nn
import math

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone
from .transformer import build_transformer
from .segmentation import build_segmentation_head


class DTFVOS(nn.Module):
    def __init__(self, backbone, transformer, segmentation_head, num_queries, train_mode, only_encoder=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_feature_levels: multi-scare features
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.num_queries = num_queries
        hidden_dim = transformer.d_model
        self.bottleneck = nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1)  # the bottleneck layer
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, transformer.nhead, dropout=0)

        self.seg_head = segmentation_head
        self.train_mode = train_mode
        self.only_encoder = only_encoder

    def load_param(self, weight):
        s = self.state_dict()
        for key, val in weight.items():
            # process ckpt from parallel module
            if key[:6] == 'module':
                key = key[7:]

            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))
        self.load_state_dict(s)

    def forward_backbone(self, frames):
        if isinstance(frames, (list, torch.Tensor)):
            frames = nested_tensor_from_tensor_list(frames)
        return self.backbone(frames)

    def extract_ref_feats(self, refer_pairs):
        frame, obj_mask = refer_pairs['frame'], refer_pairs['obj_mask']

        # TODO: fusion mask
        # concatenate object mask as a channel of image tensor
        _, no, _, _ = obj_mask.shape
        frame_mask = torch.cat([frame.expand(no, -1, -1, -1), obj_mask.transpose(0, 1).contiguous()], dim=1) # [no, 4, h, w]
        features, pos = self.forward_backbone(frame_mask)
        feat, mask = features[-1].decompose()
        pos_embed = pos[-1]

        # features, _ = self.forward_backbone(frame)
        # feat, mask = features[-1].decompose()
        
        # # filter features using segmentation mask
        # obj_mask = F.interpolate(obj_mask, size=feat.shape[-2:], mode='nearest')
        # # TODO: mask utilization
        # feat = feat * obj_mask.transpose(0, 1) + feat  # [1, c, h, w] * [no, 1, h, w] -> [no, c, h, w]
        # mask = mask.repeat(obj_mask.shape[1], 1, 1)

        # pos_embed = self.backbone[1](NestedTensor(feat, mask)).to(feat.dtype)
        return self.adjust(feat, mask, pos_embed)

    def extract_seg_feats(self, seg_frames):
        features, pos = self.forward_backbone(seg_frames)
        feat, mask = features[-1].decompose()
        pos_embed = pos[-1]

        seq_dict = self.adjust(feat, mask, pos_embed)

        median_layers = list()
        for i in range(len(features)):
            median_layers.append(features[i].decompose()[0])  # [layer1, layer2， layer3]
        return seq_dict, median_layers

    def adjust(self, feat, mask, pos_embed):
        """
        reduce channel and ajust shapes
        """
        # reduce channel
        feat = self.bottleneck(feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed.flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    def forward_transformer(self, seq_dict):
        if not self.only_encoder:
            # Forward the transformer encoder and decoder
            output_embed, enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
                                                    seq_dict["pos"], return_encoder_output=True)
            return output_embed, enc_mem
        else:
            enc_mem = self.transformer(seq_dict["feat"], seq_dict["mask"], self.query_embed.weight,
                                                    seq_dict["pos"], mode='encoder')
            return None, enc_mem

    def concat_features(self, inp_list, expand=False):
        if not expand:
            return {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
                "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
                "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}
        else:
            x, y = inp_list
            HW_x, no, C = x["feat"].shape
            HW_y, bs, _ = y["feat"].shape
            bs_no = int(bs * no)
            x_feat = x["feat"].unsqueeze(1).expand(-1, bs, -1, -1).reshape(HW_x, bs_no, C)
            y_feat = y["feat"].unsqueeze(2).expand(-1, -1, no, -1).reshape(HW_y, bs_no, C)

            x_mask = x["mask"].unsqueeze(0).expand(bs, -1, -1).reshape(bs_no, HW_x)
            y_mask = y["mask"].unsqueeze(1).expand(-1, no, -1).reshape(bs_no, HW_y)

            x_pos = x["pos"].unsqueeze(1).expand(-1, bs, -1, -1).reshape(HW_x, bs_no, C)
            y_pos = y["pos"].unsqueeze(2).expand(-1, -1, no, -1).reshape(HW_y, bs_no, C)
            return {
                "feat" : torch.cat([x_feat, y_feat], dim=0),  # H1W1+H2W2xbs*noxC
                "mask": torch.cat([x_mask, y_mask], dim=1),  # bs*noxH1W1+H2W2
                "pos": torch.cat([x_pos, y_pos], dim=0)  # H1W1+H2W2xbs*noxC
            }
            
    def segment(self, hs, mem, seq_dict, fpns, ref_mask):
        # adjust shape
        bs, _, h, w = fpns[-1].shape  # layer3
        feat_len_s = int(h*w)
        _, bs_no, C = mem.shape
        enc_opt = mem[-feat_len_s:].permute(1, 2, 0).contiguous().view(bs_no, C, h, w)
        if not self.only_encoder:
            dec_opt = hs.squeeze(0)  # (B, Q, C)

            mask = seq_dict['mask'][:, -feat_len_s:].view(bs_no, h, w)
            bbox_mask = self.bbox_attention(dec_opt, enc_opt, mask)  # (b, q, n, h, w)

            _, _, nhead, _, _ = bbox_mask.shape
            bbox_mask = bbox_mask.view(-1, nhead, h, w)
            opt_feat = torch.cat([enc_opt, bbox_mask], dim=1)
        else:
            opt_feat = enc_opt

        # enc_opt = mem[-feat_len_s:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
        # dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
        # att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
        # opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
        # bs_no, _, C, _ = opt.size()
        # opt_feat = opt.view(-1, C, h, w) # (bs*no, C, H, W)

        # multi-scale features
        no = int(bs_no / bs)
        r1, r2, _ = fpns
        r2_size, r1_size = r2.shape, r1.shape
        r2 = r2.unsqueeze(1).expand(-1, no, -1, -1, -1).reshape(bs_no, *r2_size[1:])
        r1 = r1.unsqueeze(1).expand(-1, no, -1, -1, -1).reshape(bs_no, *r1_size[1:])

        spatial_shape = (bs,) + ref_mask.shape[1:]

        logits = self.seg_head(opt_feat, r2, r1, spatial_shape)
        return logits

    def forward(self, frames, obj_masks, n_objs):
        """ The forward expects a NestedTensor, which consists of:
               - frames: batched clips, of shape [N x T x 3 x H x W]
               - masks: binary object masks of shape [N x T x M x H x W]
               - n_objs: number of objects in the video clip, [N]
            It returns a dict with the following elements:
        """
        batch_out = list()
        N = frames.shape[0]
        if self.train_mode == 'normal':
            for i in range(N):
                seq, seq_masks, n_obj = frames[i], obj_masks[i], n_objs[i]
                ref_pairs = {
                    'frame': seq[0:1],  # [1 x 3 x H x W]
                    'obj_mask': seq_masks[0:1, 1:n_obj] # [1 x no x H x W]
                }
                seg_frames = seq[1:]  # [T-1 x 3 x H x W]

                ref_feats_dict = self.extract_ref_feats(ref_pairs)
                seg_feats_dict, median_layers = self.extract_seg_feats(seg_frames)

                seq_dict = self.concat_features([ref_feats_dict, seg_feats_dict], expand=True)

                hs, enc_mem = self.forward_transformer(seq_dict)

                logits = self.segment(hs, enc_mem, seq_dict, median_layers, seq_masks[0:1])
                out = torch.softmax(logits, dim=1)  # [T-1, M, H, W]
                batch_out.append(out)
        elif self.train_mode == 'recurrent':
            for i in range(N):
                seq, seq_masks, n_obj = frames[i], obj_masks[i], n_objs[i]
                T = seq.shape[0]
                ref_feats_dict = dict()
                tmp_out = list()
                for t in range(1, T):         
                    ref_pairs = {
                        'frame': seq[t-1:t],  # [1 x 3 x H x W]
                        'obj_mask': seq_masks[0:1, 1:n_obj] if t==1 else out[:, 1:n_obj]# [1 x no x H x W]
                    }
                    seg_frame = seq[t:t+1]  # [1 x 3 x H x W]
                    if t == 1:
                        ref_feats_dict = self.extract_ref_feats(ref_pairs)
                    else:
                        ref_feats_dict = self.concat_features([ref_feats_dict, self.extract_ref_feats(ref_pairs)])
                    seg_feats_dict, median_layers = self.extract_seg_feats(seg_frame)
                    seq_dict = self.concat_features([ref_feats_dict, seg_feats_dict], expand=True)

                    hs, enc_mem = self.forward_transformer(seq_dict)

                    logits = self.segment(hs, enc_mem, seq_dict, median_layers, seq_masks[0:1])
                    out = torch.softmax(logits, dim=1)  # [1, M, H, W]
                    tmp_out.append(out)
                batch_out.append(torch.cat(tmp_out, dim=0))
        else:
            raise NotImplementedError()

        batch_out = torch.stack(batch_out, dim=0) # [N, T-1, M, H, W]
        return batch_out


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""
    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask=None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view_as(weights)
        weights = self.dropout(weights)
        return weights

    
def build_dtfvos(cfg):
    backbone = build_backbone(cfg)  # backbone and positional encoding are built together
    transformer = build_transformer(cfg)
    segmentation_head = build_segmentation_head(cfg)
    model = DTFVOS(
        backbone,
        transformer,
        segmentation_head,
        cfg.MODEL.NUM_OBJECT_QUERIES,
        cfg.TRAIN.MODE,
        cfg.MODEL.TRANSFORMER.ONLY_ENCODER
    )

    return model
