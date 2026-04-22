import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import random
import numpy as np
from mamba_ssm import Mamba


def build_valid_mask(item_seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
	idxs = torch.arange(max_len, device=item_seq_len.device)[None, :]
	return idxs < item_seq_len[:, None]


class ProjectionHead(nn.Module):
	def __init__(self, in_dim: int, hid_dim: int = 256, out_dim: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hid_dim),
			nn.BatchNorm1d(hid_dim),
			nn.GELU(),
			nn.Linear(hid_dim, out_dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


class SymmetricInfoNCE(nn.Module):
	def __init__(self, temperature: float = 0.07):
		super().__init__()
		self.temperature = temperature

	def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
		z1 = F.normalize(z1, dim=1)
		z2 = F.normalize(z2, dim=1)
		logits = torch.matmul(z1, z2.T) / self.temperature
		labels = torch.arange(z1.size(0), device=z1.device)
		loss_12 = F.cross_entropy(logits, labels)
		loss_21 = F.cross_entropy(logits.T, labels)
		return 0.5 * (loss_12 + loss_21)


class DataAugmentation:
	def __init__(
		self,
		mask_prob: float,
		span_max_len: int,
		scale_range,
		noise_std: float,
		mixup_alpha: float,
		mixup_prob: float,
		sim_threshold: float,
		enable_mixup_epoch: int,
		strong_aug_types,
		strong_aug_num: int,
	):
		self.mask_prob = float(mask_prob)
		self.span_max_len = max(1, int(span_max_len))
		self.scale_range = tuple(scale_range)
		self.noise_std = float(noise_std)
		self.mixup_alpha = float(mixup_alpha)
		self.mixup_prob = float(mixup_prob)
		self.sim_threshold = float(sim_threshold)
		self.current_epoch = 0
		self.enable_mixup_epoch = int(enable_mixup_epoch)
		self.strong_aug_types = list(strong_aug_types)
		self.strong_aug_num = int(strong_aug_num)

	def update_epoch(self, epoch: int):
		self.current_epoch = int(epoch)

	@staticmethod
	def _apply_on_valid(emb: torch.Tensor, valid_mask: torch.Tensor, value_on_invalid: float = 0.0) -> torch.Tensor:
		mask = valid_mask[..., None].float()
		return emb * mask + (1.0 - mask) * value_on_invalid

	def random_scale(self, emb: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
		scale = (
			torch.rand(emb.shape[0], 1, 1, device=emb.device)
			* (self.scale_range[1] - self.scale_range[0])
			+ self.scale_range[0]
		)
		return self._apply_on_valid(emb * scale, valid_mask, 0.0)

	def random_noise(self, emb: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
		noise = torch.randn_like(emb) * self.noise_std
		return self._apply_on_valid(emb + noise, valid_mask, 0.0)

	def span_mask(self, emb: torch.Tensor, valid_mask: torch.Tensor, drop_prob: float) -> torch.Tensor:
		B, L, H = emb.shape
		keep = valid_mask.clone()
		valid_len = valid_mask.sum(dim=1)
		target_span = (valid_len.float() * float(drop_prob)).clamp(min=1.0)
		safe_span_max = max(1, int(self.span_max_len))
		target_span = torch.minimum(target_span.floor().to(torch.long), torch.full_like(valid_len, safe_span_max))

		idxs = torch.arange(L, device=emb.device)[None, :].expand(B, L)
		valid_positions = torch.where(valid_mask, idxs, torch.full_like(idxs, L))
		valid_positions_sorted, _ = torch.sort(valid_positions, dim=1)
		max_start_in_valid = (valid_len - target_span).clamp(min=0)
		rand_in_valid = torch.rand(B, device=emb.device) * (max_start_in_valid.float() + 1e-6)
		start_offset = rand_in_valid.floor().to(torch.long)

		start_abs = torch.gather(valid_positions_sorted, 1, start_offset.unsqueeze(1)).squeeze(1)
		end_offset = start_offset + target_span - 1
		end_abs = torch.gather(valid_positions_sorted, 1, end_offset.unsqueeze(1)).squeeze(1)

		arange_L = torch.arange(L, device=emb.device)[None, :].expand(B, L)
		span_bool = (arange_L >= start_abs[:, None]) & (arange_L <= end_abs[:, None])
		keep = torch.where(span_bool, torch.zeros_like(keep, dtype=keep.dtype), keep)
		return emb * keep[..., None].float()

	def _compute_sequence_repr(self, emb: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
		mask = valid_mask[..., None].float()
		summed = (F.normalize(emb, dim=-1) * mask).sum(dim=1)
		lengths = mask.sum(dim=1).clamp(min=1.0)
		return summed / lengths

	def semantic_mixup(self, emb: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
		B, L, H = emb.shape
		with torch.no_grad():
			seq_repr_raw = self._compute_sequence_repr(emb, valid_mask)
			seq_repr_dir = F.normalize(seq_repr_raw, dim=-1)
			sim = torch.matmul(seq_repr_dir, seq_repr_dir.T)
			sim.fill_diagonal_(-1e9)
			sim_vals, nn_idx = sim.max(dim=1)

			norms = seq_repr_raw.norm(p=2, dim=-1)
			partner_norms = norms[nn_idx]
			min_norm = torch.minimum(norms, partner_norms)
			mu = norms.mean().clamp(min=1e-6)
			beta = 0.5
			norm_ratio = (min_norm / mu).pow(beta)
			alpha = 4.0
			g = torch.sigmoid(alpha * (norm_ratio - 1.0))

		beta = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
		lam = beta.sample((B,)).to(emb.device)

		w_sim = (sim_vals + 1.0) * 0.5
		mix_weight = torch.clamp(w_sim * g, 0.0, 1.0)
		lam = (1.0 - mix_weight) * lam + mix_weight * 0.5

		partner = emb[nn_idx]
		lam_b = lam.view(B, 1, 1)
		mixed = lam_b * emb + (1.0 - lam_b) * partner
		return self._apply_on_valid(mixed, valid_mask, 0.0)

	def augment(self, emb: torch.Tensor, item_seq_len: torch.Tensor):
		valid_mask = build_valid_mask(item_seq_len, emb.size(1))

		selected1 = self.strong_aug_types
		if len(self.strong_aug_types) > 0:
			k1 = min(self.strong_aug_num, len(self.strong_aug_types))
			selected1 = random.sample(self.strong_aug_types, k1)

		aug_weak = emb
		if 'noise' in selected1:
			aug_weak = self.random_noise(aug_weak, valid_mask)
		if 'scale' in selected1:
			aug_weak = self.random_scale(aug_weak, valid_mask)
		if 'mask' in selected1:
			aug_weak = self.span_mask(aug_weak, valid_mask, drop_prob=self.mask_prob)
		if 'semantic_mixing' in selected1:
			if self.current_epoch >= self.enable_mixup_epoch:
				aug_weak = self.semantic_mixup(aug_weak, valid_mask)

		selected = self.strong_aug_types
		if len(self.strong_aug_types) > 0:
			k = min(self.strong_aug_num, len(self.strong_aug_types))
			selected = random.sample(self.strong_aug_types, k)

		aug_strong = emb
		if 'noise' in selected:
			aug_strong = self.random_noise(aug_strong, valid_mask)
		if 'scale' in selected:
			aug_strong = self.random_scale(aug_strong, valid_mask)
		if 'mask' in selected:
			aug_strong = self.span_mask(aug_strong, valid_mask, drop_prob=self.mask_prob)
		if 'semantic_mixing' in selected:
			if self.current_epoch >= self.enable_mixup_epoch:
				aug_strong = self.semantic_mixup(aug_strong, valid_mask)

		return aug_weak, aug_strong


class FiCoRec(SequentialRecommender):
	def __init__(self, config, dataset):
		super(FiCoRec, self).__init__(config, dataset)
		self.config = config

		self.hidden_size = config["hidden_size"]
		self.loss_type = config["loss_type"]
		self.num_layers = config["num_layers"]
		self.dropout_prob = config["dropout_prob"]

		self.cl_weight = config["cl_weight"]
		self.temperature = config["temperature"]

		self.d_state = config["d_state"]
		self.d_conv = config["d_conv"]
		self.expand = config["expand"]

		self.item_embedding = nn.Embedding(
			self.n_items, self.hidden_size, padding_idx=0
		)

		self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
		self.dropout = nn.Dropout(self.dropout_prob)
		self._tail_score = nn.Linear(self.hidden_size, 1)
		self.attn_recency_alpha = nn.Parameter(torch.tensor(0.3))
		self._global_query = nn.Parameter(torch.randn(self.hidden_size) * 0.02)
		self._fuse_gate = nn.Linear(self.hidden_size * 2, 1)

		self.mamba_layers = nn.ModuleList([
			MambaLayer(
				d_model=self.hidden_size,
				d_state=self.d_state,
				d_conv=self.d_conv,
				expand=self.expand,
				dropout=self.dropout_prob,
				num_layers=self.num_layers,
			) for _ in range(self.num_layers)
		])

		mask_prob = config["mask_prob"]
		span_max_len = config["span_max_len"]
		scale_range = tuple(config["scale_range"])
		noise_std = config["noise_std"]
		mixup_alpha = config["mixup_alpha"]
		mixup_prob = config["mixup_prob"]
		sim_threshold = config["sim_threshold"]
		enable_mixup_epoch = config["enable_mixup_epoch"]
		aug_types = list(config["aug_types"])
		aug_num = int(config["aug_num"])

		self.data_aug = DataAugmentation(
			mask_prob=mask_prob,
			span_max_len=span_max_len,
			scale_range=scale_range,
			noise_std=noise_std,
			mixup_alpha=mixup_alpha,
			mixup_prob=mixup_prob,
			sim_threshold=sim_threshold,
			enable_mixup_epoch=enable_mixup_epoch,
			strong_aug_types=aug_types,
			strong_aug_num=aug_num,
		)

		proj_hid = config["proj_hid"]
		proj_out = config["proj_out"]
		self.proj_head = ProjectionHead(self.hidden_size, hid_dim=proj_hid, out_dim=proj_out)

		# weights for multi-branch contrastive learning (sequence / tail / global)
		if "cl_seq_weight" in config:
			self.cl_seq_weight = float(config["cl_seq_weight"])
		else:
			self.cl_seq_weight = 1.0

		if "cl_tail_weight" in config:
			self.cl_tail_weight = float(config["cl_tail_weight"])
		else:
			self.cl_tail_weight = 0.5

		if "cl_global_weight" in config:
			self.cl_global_weight = float(config["cl_global_weight"])
		else:
			self.cl_global_weight = 0.2

		self.cl_loss = SymmetricInfoNCE(temperature=self.temperature)

		if self.loss_type == "BPR":
			self.loss_fct = BPRLoss()
		elif self.loss_type == "CE":
			self.loss_fct = nn.CrossEntropyLoss()
		else:
			raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def forward(self, item_seq, item_seq_len, return_emb=False):
		item_emb = self.item_embedding(item_seq)
		item_emb = self.dropout(item_emb)
		item_emb = self.LayerNorm(item_emb)

		for i in range(self.num_layers):
			item_emb = self.mamba_layers[i](item_emb)

		seq_output = self.aggregate_sequence(item_emb, item_seq_len)

		if return_emb:
			return seq_output, item_emb
		return seq_output

	def update_epoch(self, epoch):
		self.data_aug.update_epoch(epoch)

	def calculate_loss(self, interaction):
		item_seq = interaction[self.ITEM_SEQ]
		item_seq_len = interaction[self.ITEM_SEQ_LEN]

		item_emb = self.item_embedding(item_seq)
		item_emb = self.dropout(item_emb)
		item_emb = self.LayerNorm(item_emb)

		with torch.no_grad():
			aug_emb1, aug_emb2 = self.data_aug.augment(item_emb, item_seq_len)

		# obtain tail/global/sequence representations for each augmented view
		tail_vec1, global_vec1, aug_seq1 = self.encode_sequence_components(aug_emb1, item_seq_len)
		tail_vec2, global_vec2, aug_seq2 = self.encode_sequence_components(aug_emb2, item_seq_len)

		for i in range(self.num_layers):
			item_emb = self.mamba_layers[i](item_emb)
		seq_output = self.aggregate_sequence(item_emb, item_seq_len)

		pos_items = interaction[self.POS_ITEM_ID]
		if self.loss_type == "BPR":
			neg_items = interaction[self.NEG_ITEM_ID]
			pos_items_emb = self.item_embedding(pos_items)
			neg_items_emb = self.item_embedding(neg_items)
			pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
			neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
			main_loss = self.loss_fct(pos_score, neg_score)
		else:
			test_item_emb = self.item_embedding.weight
			logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
			main_loss = self.loss_fct(logits, pos_items)

		if self.cl_weight > 0:
            #使用投影头
			z_seq1 = self.proj_head(aug_seq1)
			z_seq2 = self.proj_head(aug_seq2)
			z_tail1 = self.proj_head(tail_vec1)
			z_tail2 = self.proj_head(tail_vec2)
			z_global1 = self.proj_head(global_vec1)
			z_global2 = self.proj_head(global_vec2)
            
            #不使用投影头
			# z_seq1 = aug_seq1
			# z_seq2 = aug_seq2
			# z_tail1 = tail_vec1
			# z_tail2 = tail_vec2
			# z_global1 = global_vec1
			# z_global2 = global_vec2

			cl_seq = self.cl_loss(z_seq1, z_seq2)
			cl_tail = self.cl_loss(z_tail1, z_tail2)
			cl_global = self.cl_loss(z_global1, z_global2)

			cl_loss = (
				self.cl_seq_weight * cl_seq
				+ self.cl_tail_weight * cl_tail
				+ self.cl_global_weight * cl_global
			)
			total_loss = main_loss + self.cl_weight * cl_loss
		else:
			total_loss = main_loss

		return total_loss

	def encode_sequence(self, item_emb, item_seq_len):
		item_emb = self.dropout(item_emb)
		item_emb = self.LayerNorm(item_emb)

		for i in range(self.num_layers):
			item_emb = self.mamba_layers[i](item_emb)

		return self.aggregate_sequence(item_emb, item_seq_len)

	def encode_sequence_components(self, item_emb, item_seq_len):
		"""
		Encode sequence and return tail/global/sequence-level representations
		for use in multi-branch contrastive learning.
		"""
		item_emb = self.dropout(item_emb)
		item_emb = self.LayerNorm(item_emb)

		for i in range(self.num_layers):
			item_emb = self.mamba_layers[i](item_emb)

		tail_vec, global_vec, seq_output = self._aggregate_components(item_emb, item_seq_len)
		return tail_vec, global_vec, seq_output

	def aggregate_sequence(self, hidden_seq: torch.Tensor, item_seq_len: torch.Tensor) -> torch.Tensor:
		_, _, seq_output = self._aggregate_components(hidden_seq, item_seq_len)
		return seq_output

	def _aggregate_components(self, hidden_seq: torch.Tensor, item_seq_len: torch.Tensor):
		use_tail_pool = bool(self.config["use_tail_pool"])
		tail_k = int(self.config["tail_k"])

		B, L, H = hidden_seq.shape
		valid_mask = build_valid_mask(item_seq_len, L)

		if not use_tail_pool:
			tail_vec = self.gather_indexes(hidden_seq, item_seq_len - 1)
		else:
			arange_L = torch.arange(L, device=hidden_seq.device)[None, :].expand(B, L)
			last_idx = (item_seq_len - 1).clamp(min=0)[:, None]
			start_idx = (item_seq_len - tail_k).clamp(min=0)[:, None]
			tail_mask = (arange_L >= start_idx) & (arange_L <= last_idx)
			attn_mask = valid_mask & tail_mask

			logits_tail = self._tail_score(hidden_seq).squeeze(-1)
			positions = torch.arange(L, device=hidden_seq.device).float()
			dist_from_tail = (L - 1) - positions
			alpha = F.softplus(self.attn_recency_alpha)
			logits_tail = logits_tail + (-alpha) * dist_from_tail.view(1, L)

			logits_tail = logits_tail.masked_fill(~attn_mask, float('-inf'))
			attn_tail = torch.softmax(logits_tail, dim=1)
			attn_tail = attn_tail.masked_fill(~attn_mask, 0.0)
			attn_tail = attn_tail / (attn_tail.sum(dim=1, keepdim=True) + 1e-6)
			tail_vec = torch.bmm(attn_tail.unsqueeze(1), hidden_seq).squeeze(1)
			zero_mask = (attn_tail.sum(dim=1) <= 0)
			if zero_mask.any():
				fallback = self.gather_indexes(hidden_seq, item_seq_len - 1)
				tail_vec = torch.where(zero_mask[:, None], fallback, tail_vec)

		q = F.normalize(self._global_query, dim=0)
		logits_global = torch.matmul(hidden_seq, q)
		logits_global = logits_global.masked_fill(~valid_mask, float('-inf'))
		attn_global = torch.softmax(logits_global, dim=1)
		attn_global = attn_global.masked_fill(~valid_mask, 0.0)
		attn_global = attn_global / (attn_global.sum(dim=1, keepdim=True) + 1e-6)
		global_vec = torch.bmm(attn_global.unsqueeze(1), hidden_seq).squeeze(1)
		zero_mask_g = (attn_global.sum(dim=1) <= 0)
		if zero_mask_g.any():
			fallback = self.gather_indexes(hidden_seq, item_seq_len - 1)
			global_vec = torch.where(zero_mask_g[:, None], fallback, global_vec)

		fuse_inp = torch.cat([tail_vec, global_vec], dim=-1)
		gate = torch.sigmoid(self._fuse_gate(fuse_inp))
		seq_output = gate * tail_vec + (1.0 - gate) * global_vec
		return tail_vec, global_vec, seq_output

	def predict(self, interaction):
		item_seq = interaction[self.ITEM_SEQ]
		item_seq_len = interaction[self.ITEM_SEQ_LEN]
		test_item = interaction[self.ITEM_ID]
		seq_output = self.forward(item_seq, item_seq_len)
		test_item_emb = self.item_embedding(test_item)
		scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
		return scores

	def full_sort_predict(self, interaction):
		item_seq = interaction[self.ITEM_SEQ]
		item_seq_len = interaction[self.ITEM_SEQ_LEN]
		seq_output = self.forward(item_seq, item_seq_len)
		test_items_emb = self.item_embedding.weight
		scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
		return scores

class MambaLayer(nn.Module):
	def __init__(self, d_model, d_state, d_conv, expand, dropout, num_layers):
		super().__init__()
		self.num_layers = num_layers
		self.mamba = Mamba(
				d_model=d_model,
				d_state=d_state,
				d_conv=d_conv,
				expand=expand,
			)
		self.dropout = nn.Dropout(dropout)
		self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
		self.ffn = FeedForward(d_model=d_model, inner_size=d_model*4, dropout=dropout)

	def forward(self, input_tensor):
		hidden_states = self.mamba(input_tensor)
		if self.num_layers == 1:
			hidden_states = self.LayerNorm(self.dropout(hidden_states))
		else:
			hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
		hidden_states = self.ffn(hidden_states)
		return hidden_states


class FeedForward(nn.Module):
	def __init__(self, d_model, inner_size, dropout=0.2):
		super().__init__()
		self.w_1 = nn.Linear(d_model, inner_size)
		self.w_2 = nn.Linear(inner_size, d_model)
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)

	def forward(self, input_tensor):
		hidden_states = self.w_1(input_tensor)
		hidden_states = self.activation(hidden_states)
		hidden_states = self.dropout(hidden_states)

		hidden_states = self.w_2(hidden_states)
		hidden_states = self.dropout(hidden_states)
		hidden_states = self.LayerNorm(hidden_states + input_tensor)

		return hidden_states

