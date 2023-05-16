import gc
import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def to_id_count_list(id_list):
    current = None
    count = 0
    results = []
    for next_id in id_list + [None]:
        if next_id != current:
            if current is not None:
                results.append((current, count))
            count = 1
            current = next_id
        else:
            count += 1
    return results


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def gen_groups_asic_2d_extended(in_size, keep_k, dtype='int8', asic_input_gloup=8, cgb=512, group_size_value=64):
    input_channel_max_num = 8 * cgb if dtype == 'int8' else 4 * cgb
    results = []
    for i in range(math.ceil(in_size / input_channel_max_num)):
        start_index = i * input_channel_max_num
        end_index = min(in_size, (i + 1) * input_channel_max_num)
        sub_results = gen_groups_asic_2d(
            in_size=end_index - start_index,
            keep_k=int(keep_k / in_size * (end_index - start_index)),
            dtype=dtype,
            asic_input_gloup=asic_input_gloup,
            cgb=cgb,
            group_size_value=group_size_value)
        sub_results = [(r[0] + start_index, r[1]) for r in sub_results]
        results.extend(sub_results)
    return results


def gen_groups_asic_2d(in_size, keep_k, dtype='int8', asic_input_gloup=8, cgb=512, group_size_value=64):
    def _block_sparsity_balance(transpose_weight, keep_k):
        reshape_weight = transpose_weight
        base_k = keep_k // reshape_weight.shape[0]   # OI
        remain_k = keep_k % reshape_weight.shape[0]
        if remain_k > 0:
            num_keep = min(reshape_weight.shape[-1], base_k + 1)
        else:
            num_keep = min(reshape_weight.shape[-1], base_k)
        return reshape_weight[0], num_keep

    def _block_1x1(transpose_weight, keep_k, asic_input_gloup=8):
        results = []
        temp1 = transpose_weight.shape[-1] // asic_input_gloup
        lists = [[] for _ in range(asic_input_gloup)]
        for i in range(temp1):
            for j in range(i * asic_input_gloup, (i + 1) * asic_input_gloup):
                lists[j % asic_input_gloup].append(j)
        for i in range(temp1 * asic_input_gloup, transpose_weight.shape[-1]):
            lists[i % asic_input_gloup].append(i)
        temp3 = []
        for i in range(asic_input_gloup):
            value = int(len(lists[i]) / transpose_weight.shape[-1] * keep_k)
            temp3.append(max(value, 1))
        for i in range(asic_input_gloup):
            temp4 = np.concatenate([transpose_weight[:, one: one + 1] for one in lists[i]], 1)
            results.append(_block_sparsity_balance(temp4, temp3[i]))
        return results

    def find_valid_index(ids, in_size):
        for idx, tensor_idx in enumerate(ids):
            if tensor_idx >= in_size:
                return idx

    def computer_mask(weight, in_size, group_size, block_size, keep_k, patch_size, asic_input_gloup):
        temp1_1 = max(int(np.ceil(in_size / group_size)), 1)
        ids0 = np.arange(block_size)
        ids0 = np.concatenate([ids0 + (row_id * group_size) for row_id in range(temp1_1)])
        keep_k0 = int(block_size * temp1_1 * keep_k / in_size)
        results = []
        for col_id in range(patch_size):  # cpart
            ids = ids0 + (block_size * col_id)
            temp_v = len(ids)
            if min(ids) >= in_size: break
            if len(ids) > in_size: ids = ids[:in_size]
            ids = ids[:find_valid_index(ids, in_size)] if in_size-1 < max(ids) else ids
            cur_k = int((len(ids) / temp_v) * keep_k0)
            results.extend(_block_1x1(
                np.transpose(weight[ids, :], [1, 0]).astype(dtype=weight.dtype),  # IO --> OI
                cur_k,
                asic_input_gloup
            ))
        return results

    group_size_max = None
    group_size = None
    block_size = None
    if dtype in ['bf16', 'bfloat16']:
        group_size_max = 256
        group_size = cgb
        block_size = group_size_value // 2
    elif dtype == 'int8':
        group_size_max = 512
        group_size = cgb
        block_size = group_size_value
    weight = np.arange(in_size).reshape(-1, 1)
    in_size, out_size = weight.shape
    if group_size > block_size:
        assert group_size % block_size == 0
    patch_size = max(group_size // block_size, 1)
    results = []
    if (in_size / patch_size) > group_size_max:
        ori_insize = in_size
        ori_keep_k = keep_k
        inc_group_size = int(np.ceil(in_size / group_size_max))
        for i in range(inc_group_size):
            weight_group = weight[i * group_size_max:(i + 1) * group_size_max, :]
            in_size, out_size = weight_group.shape
            keep_k = ori_keep_k * (in_size / ori_insize)
            results.extend(
                computer_mask(weight_group, in_size, group_size, block_size, keep_k, patch_size, asic_input_gloup))
    else:
        if in_size < block_size:
            block_size = in_size
        if group_size < group_size_value and dtype == 'int8':
            group_size = group_size_value
        if group_size < (group_size_value // 2) and dtype == 'bf16':
            group_size = group_size_value // 2

        results.extend(computer_mask(weight, in_size, group_size, block_size, keep_k, patch_size, asic_input_gloup))
    return results


class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()


class MoffettSolver(SparseGPT):
    def invert(self, H: torch.Tensor):
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + 1e-4 * torch.eye(H.shape[0], device=self.device)))
        except RuntimeError:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H + torch.eye(H.shape[0], device=self.device)))
        return Hinv
    
class ABCSolver(MoffettSolver):
    def prune_structured(
        self, sparsity, prunen=0, prunem=0, 
        blocksize=128, percdamp=.01, verbose=True,
        cgb=512, groups=8, perm_score=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        gc.collect()

        ''' -------- Start from ABC Solver ------------ '''

        columns = W.shape[1]

        # group permutation ori
        # original_index = torch.arange(columns)
        # group_ids = (torch.div(original_index, cgb, rounding_mode='trunc') * groups + original_index % groups + 1.0)
        # group permutation new
        group_ids_list = gen_groups_asic_2d_extended(
            in_size=columns, keep_k=int(columns * (1.0 - sparsity)), cgb=cgb, asic_input_gloup=groups
        )
        group_ids = np.zeros(columns)
        for i, (group_index, _) in enumerate(group_ids_list):
            group_ids[group_index] = i
        group_ids = torch.from_numpy(group_ids)
        
        perm = torch.argsort(group_ids)
        W = W[:, perm]
        H = H[:, perm][perm, :]
        group_ids = group_ids[perm]

        # score permutation
        if perm_score:
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0
            Hinv = self.invert(H)
            diag = torch.diagonal(Hinv)
            scores = torch.sum(((W ** 2) / diag), dim=0)
            score_set = set()
            cum_index = 0
            for i, group_size in to_id_count_list(group_ids.tolist()):
                group_score = float(torch.mean(scores[cum_index: cum_index+group_size]).cpu().numpy())
                while group_score in score_set:
                    group_score += 1e-9
                score_set.add(group_score)
                scores[cum_index: cum_index + group_size] = group_score
                cum_index += group_size
            perm_2 = torch.argsort(scores)
            W = W[:, perm_2]
            H = H[:, perm_2][perm_2, :]
            group_ids = group_ids[perm_2]
        
        ''' -------- End of ABC Solver ------------ '''
        
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)

        Hinv = H
        del H
        gc.collect()

        ''' -------- Start from ABC Solver ------------ '''

        i1 = 0
        for _, group_size in to_id_count_list(group_ids.tolist()):
            i2 = min(i1 + group_size, columns)
            count = i2 - i1
            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            scores = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
            mask1 = (torch.argsort(torch.argsort(scores, dim=1), dim=1) < int(group_size * sparsity)).bool()
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]
                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
            i1 += group_size

        torch.cuda.synchronize()
        if perm_score:
            W = W[:, torch.argsort(perm_2)]
        W = W[:, torch.argsort(perm)]
        print(torch.mean((W == 0.0).float()))
        ''' -------- End of ABC Solver ------------ '''

        torch.cuda.synchronize()
        if verbose:
            print('time used: %.2f' % (time.time() - tick))

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        del W
        gc.collect()

        if DEBUG:
            if verbose:
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

class OBCSolver(MoffettSolver):
    def prepare_iter(self, i1, parallel, W, device):
        i2 = min(i1 + parallel, W.shape[0])
        count = i2 - i1
        w = W[i1:i2, :]
        mask = torch.zeros_like(w, device=W.device).bool()
        range_count = torch.arange(count, device=W.device)
        return i2, w, mask, range_count

    def prepare_sparse(self, w, mask, Hinv):
        start = int(torch.min(torch.sum((w == 0).float(), 1)).item()) + 1
        for i in range(w.shape[0]):
            tmp = w[i] == 0
            H1 = Hinv[i]
            H1[tmp, :] = 0
            H1[:, tmp] = 0
            H1[tmp, tmp] = 1
            Hinv[i] = self.invert(H1)
            mask[i, torch.nonzero(tmp, as_tuple=True)[0][:(start - 1)]] = True
        return start

    def trim(self, w, H, num_keep):
        trim_w = torch.zeros(w.shape[0], num_keep, device=H.device)
        trim_mask = torch.zeros_like(trim_w, device=H.device).bool()
        trim_H = torch.zeros(H.shape[0], num_keep, num_keep, device=H.device)
        indicators = torch.ones_like(w, device=H.device).bool()
        for i in range(w.shape[0]):
            ind = torch.argsort(torch.argsort(-torch.abs(w[i]))) < num_keep
            indicators[i] = ind
            trim_w[i] = w[i][ind]
            trim_H[i] = H[i][ind][:, ind]
        return trim_w, trim_mask, trim_H, indicators

    # def prune_structured(self, W: torch.Tensor, H: torch.Tensor, parallel=128):
    def prune_structured(
        self, sparsity, prunen=0, prunem=0, 
        blocksize=128, percdamp=.01, verbose=True,
        cgb=512, groups=8, perm_score=False, parallel=128,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H

        ''' -------- Start from OBC Solver ------------ '''
        rows, columns = W.shape
        W = W.clone()
        pruned_W = W.clone()
        H = H.float()
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        group_ids_list = gen_groups_asic_2d_extended(
            in_size=columns, keep_k=int(columns * (1.0 - sparsity)), cgb=cgb, asic_input_gloup=groups
        )
        assert len(set([group_index.size for group_index, _ in group_ids_list])) == 1
        group_size = group_ids_list[0][0].size
        group_ids = np.zeros(columns)
        for i, (group_index, _) in enumerate(group_ids_list):
            group_ids[group_index] = i
        group_ids = torch.from_numpy(group_ids)

        perm = torch.argsort(group_ids)
        W = W[:, perm]
        H = H[:, perm][perm, :]

        for i1 in range(0, rows, parallel):
            i2, w, mask, range_count = self.prepare_iter(i1, parallel, W, W)
            # Hinv is initialized by H
            Hinv = H.unsqueeze(0).repeat((i2 - i1, 1, 1))

            # trim for speed up
            min_num_zeros = int(torch.min(torch.sum((w == 0).float(), 1)).item())
            trim = min_num_zeros / columns > 0.1
            if trim:
                w, mask, Hinv, indicators = self.trim(w=w, H=Hinv, num_keep=columns - min_num_zeros)
            start = self.prepare_sparse(w, mask, Hinv)
            if trim:
                start = start + min_num_zeros

            group_mask = mask.clone()
            for _ in range(start, int(columns * sparsity)):
                group_mask[:, :columns] = (torch.sum(
                    mask[:, :columns].reshape(i2 - i1, -1, group_size).float(), dim=2, keepdim=True
                ) >= (group_size * sparsity)).repeat(1, 1, group_size).reshape(i2 - i1, -1)
                diag = torch.diagonal(Hinv, dim1=1, dim2=2)
                scores = (w ** 2) / diag
                scores[mask] = float('inf')
                scores[group_mask] = float('inf')
                j = torch.argmin(scores, 1)
                row = Hinv[range_count, j, :]
                d = diag[range_count, j]
                w -= row * (w[range_count, j] / d).unsqueeze(1)
                mask[range_count, j] = True
                w[mask] = 0
                row /= torch.sqrt(d).unsqueeze(1)
                Hinv -= torch.bmm(row.unsqueeze(2), row.unsqueeze(1))

            if trim:
                for i in range(w.shape[0]):
                    pruned_W[i1 + i, indicators[i]] = w[i]
            else:
                pruned_W[i1:i2, :] = w

            # torch.cuda.synchronize()
        pruned_W = pruned_W[:, torch.argsort(perm)]

        ''' -------- End of OBC Solver ------------ '''

        torch.cuda.synchronize()
        W = pruned_W
        if verbose:
            print('time used: %.2fs' % (time.time() - tick))

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            if verbose:
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
