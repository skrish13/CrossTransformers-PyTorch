import torch
import torch.nn as nn
from resnet import resnet34

class CrossAttention(nn.Module):
	def __init__(self, config={}):
		super().__init__()

		self.feature_extractor = resnet34()

		self.config = {
			'query_feat_size': 512,
			'support_feat_size': 512,
			'key_head_dim': 128,
			'value_head_dim': 128,
		}

		D = self.config['query_feat_size']
		Ds = self.config['support_feat_size']
		dk = self.config['key_head_dim']
		dv = self.config['value_head_dim']

		self.key_head = nn.Conv2d(Ds, dk, 1, bias=False)
		self.query_head = nn.Conv2d(D, dk, 1, bias=False)
		self.value_head = nn.Conv2d(Ds, dv, 1, bias=False)

		## In the paper authors use key and query head to be same; End of Section 3.2
		## Feel free to comment if you dont prefer this
		self.query_head = self.key_head

	def forward(self, query, support):
		""" query   B x D x H x W
			support Nc x Nk x Ds x Hs x Ws (#CLASSES x #SHOT x #DIMENSIONS)
		"""

		Nc, Nk, Ds, Hs, Ws = support.shape

		### Step 1: Get query and support features
		query_image_features = self.feature_extractor(query)
		support_image_features = self.feature_extractor(support.view(Nc*Nk, Ds, Hs, Ws))


		### Step 2: Calculate query aligned prototype
		query = self.query_head(query_image_features)
		support_key = self.key_head(support_image_features)
		support_value = self.value_head(support_image_features)

		dk = query.shape[1]

		## flatten pixels in query (p in the paper)
		query = query.view(query.shape[0], query.shape[1], -1)
		
        ## flatten pixels & k-shot in support (j & m in the paper respectively)
		support_key = support_key.view(Nc, Nk, support_key.shape[1], -1)
		support_value = support_value.view(Nc, Nk, support_value.shape[1], -1)

		support_key = support_key.permute(0, 2, 3, 1)		
		support_value = support_value.permute(0, 2, 3, 1)

		support_key = support_key.view(Nc, support_key.shape[1], -1)		
		support_value = support_value.view(Nc, support_value.shape[1], -1)

		## v is j images' m pixels, ie k-shot*h*w
		attn_weights = torch.einsum('bdp,ndv->bnpv', query, support_key) * (dk ** -0.5)
		attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
		
        ## get weighted sum of support values
		support_value = support_value.unsqueeze(0).expand(attn_weights.shape[0], -1, -1, -1)
		query_aligned_prototype = torch.einsum('bnpv,bndv->bnpd', attn_weights, support_value)

		### Step 3: Calculate query value
		query_value = self.value_head(query_image_features)
		query_value = query_value.view(query_value.shape[0], -1, query_value.shape[1]) ##bpd
		
		### Step 4: Calculate distance between queries and supports
		distances = []
		for classid in range(query_aligned_prototype.shape[1]):
			dxc = torch.cdist(query_aligned_prototype[:, classid], 
											query_value, p=2)
			dxc = dxc**2
			B,P,R = dxc.shape
			dxc = dxc.sum(dim=(1,2)) / (P*R)
			distances.append(dxc)
		
		distances = torch.stack(distances, dim=1)

		return distances