import torch
import torch.nn as nn

class CrossAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, config={}):
        super().__init__()

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
            support Nc x Ds x Hs x Ws
        """

        query = self.query_head(query)
        query_value = self.value_head(query)
        support_key = self.key_head(support)
        support_value = self.value_head(support)

        dk = query.shape[1]

        ## flatten pixels for simplicity
        query = query.view(query.shape[0], query.shape[1], -1)
        query_value = query_value.view(query_value.shape[0], query_value.shape[1], -1)
        support_key = support_key.view(support_key.shape[0], support_key.shape[1], -1)
        support_value = support_value.view(support_value.shape[0], support_value.shape[1], -1)

        ## dot product over every query and key features
        scores = torch.einsum('bdu,ndv->bnuv', query, key) / dk**.5
        ## sum over support set length 'n' and all feature vectors 'v' in each support image
        attn_weights = torch.nn.functional.softmax(scores, dim=3)
        attn_weights = torch.nn.functional.softmax(scores, dim=1)
        ## sum over support set length 'n' and all feature vectors 'v' in each support image
        query_aligned = torch.einsum('bnuv,ndv->bdv', attn_weights, support_value)

        ## reshape for cdist
        query_aligned = query_aligned.permute(0,2,1)
        query_value = query_value.permute(0,2,1)     

        ## calculate scalar distance
        distance = torch.cdist(query_aligned, query_value, p=2)
        distance = distance**2
        B,P,R = distance.shape
        distance = distance.sum(dim=(1,2)) / (P*R)

        return distance
