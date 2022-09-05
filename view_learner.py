import torch
from torch.nn import Sequential, Linear, ReLU
from opt import  args
from torch import nn



class ViewLearner(torch.nn.Module):
	def __init__(self, msk_fea_dim,hid_units,mlp_edge_model_dim=64):
		super(ViewLearner, self).__init__()

		
		# self.input_dim = args.emb_dim
		self.input_dim = hid_units
		self.mlp_edge_model = Linear(self.input_dim * 2,1)
		self.mlp_fea_masking_model= Linear(self.input_dim, msk_fea_dim)
		self.init_emb()

	def init_emb(self):
		for m in self.modules():
			if isinstance(m, Linear):
				torch.nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, encoder,x,adj,edge_index):
		node_emb = encoder(x,adj)
		src, dst = edge_index[0], edge_index[1]
		# emb_src = node_emb[src]
		# emb_dst = node_emb[dst]
		# edge_logits = torch.bmm(node_emb[src].view(-1, 1, self.input_dim), node_emb[dst].view(-1, self.input_dim, 1)).squeeze()
	#	print(emb_src.shape)
		edge_emb = torch.cat([node_emb[src], node_emb[dst]], 1)
	#	print(edge_emb.shape)
		edge_logits = self.mlp_edge_model(edge_emb)
		fea_logits	= self.mlp_fea_masking_model(node_emb)
		return edge_logits,fea_logits