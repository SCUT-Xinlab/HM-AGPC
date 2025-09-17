import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

lamda = config.lamda
SE = config.SE
node = config.node  # number of rois

class HM_AG_GPC_SE(nn.Module):
	def __init__(self, in_dim, out_dim, node_list, atlas = 'aal'):
		super(HM_AG_GPC_SE, self).__init__()
		self.node_list = node_list
		self.out_dim = out_dim
		self.conv_1 = nn.Conv2d(in_dim, out_dim, (1,node))
		nn.init.normal_(self.conv_1.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))
		self.conv_2 = nn.Conv2d(in_dim, out_dim, (1, node))
		nn.init.normal_(self.conv_2.weight, std=math.sqrt(2 * (1 - lamda) / (node * in_dim + node * out_dim)))
		self.convres = nn.Conv2d(in_dim, out_dim, 1)
		nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))
		self.trans_1 = nn.Conv2d(out_dim, out_dim // 2, 1)
		nn.init.normal_(self.trans_1.weight, std=math.sqrt(4 / (out_dim + out_dim // 2)))
		self.trans_2 = nn.Conv2d(out_dim + 4, out_dim // 2, 1)
		nn.init.normal_(self.trans_2.weight, std=math.sqrt(4 / (out_dim + out_dim // 2)))

		if atlas == 'aal':
			fc_1_d = 2960
			fc_2_d = 10496
		# this is the partition prior for AAL atlas, should be changed based on the chosen atlas.

		self.fc_1 = nn.Linear(fc_1_d, 1, bias=False)
		self.fc_2 = nn.Linear(fc_2_d, 1, bias=False)
		self.sed = nn.Linear(out_dim, int(out_dim/SE), bias=False)
		self.seu = nn.Linear(int(out_dim/SE), out_dim, bias=False)

	def forward(self, x, x_c0, embedding):
		batchsize = x.shape[0]
		res = self.convres(x)
		hembed = embedding(torch.LongTensor(self.node_list).cuda()).cuda()
		hembed = hembed.unsqueeze(2).unsqueeze(3)
		hembed = hembed.expand(node, 4, batchsize, 1)
		hembed = hembed.permute(2, 1, 0, 3)

		# HOMO atten.
		q_x_cl = self.trans_1(x_c0).permute(0, 3, 2, 1)
		k_x_cl = q_x_cl.permute(0, 1, 3, 2)
		qk_x_cl = q_x_cl @ k_x_cl/math.sqrt(self.out_dim // 2)

		# HETERO atten.
		x_c0 = torch.cat((hembed, x_c0), dim=1)
		q_x = self.trans_2(x_c0).permute(0, 3, 2, 1)
		k_x = q_x.permute(0, 1, 3, 2)
		qk_x = q_x @ k_x / math.sqrt(self.out_dim // 2)

		# creates masks to identify homo part and hetero part
		node_l = torch.LongTensor(self.node_list).cuda()
		node_l_ex = node_l.view(1, node, 1).expand(batchsize, node, node)
		mask_homo = (node_l_ex == node_l_ex.transpose(1, 2)) 	# homo part
		mask_hetero = ~mask_homo											# hetero part

		# get homo base vector and hetero base vector
		qk_x_cl_flat = qk_x_cl.squeeze(1)
		qk_x_flat = qk_x.squeeze(1)
		A = (qk_x_cl_flat * mask_homo.float()).view(batchsize, -1)
		B = (qk_x_flat * mask_hetero.float()).view(batchsize, -1)
		A_nonz = torch.nonzero(A[0], as_tuple=False)
		B_nonz = torch.nonzero(B[0], as_tuple=False)
		homo = torch.index_select(A, dim=1, index=A_nonz.squeeze()) 	# homo base vector in homo atten.
		hetero = torch.index_select(B, dim=1, index=B_nonz.squeeze())	# hetero base vector in hetero atten.

		# get homo threshold and hetero threshold
		base_homo = self.fc_1(homo)
		base_hetero = self.fc_2(hetero)

		# prepare for masking
		base_homo_ex = base_homo.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, node, node)
		base_hetero_ex = base_hetero.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, node, node)
		mask_homo_ex = mask_homo.unsqueeze(-1).permute(0, 3, 1, 2)
		mask_hetero_ex = mask_hetero.unsqueeze(-1).permute(0, 3, 1, 2)
		total_base = mask_homo_ex * base_homo_ex + mask_hetero_ex * base_hetero_ex
		qk_x = qk_x.expand(batchsize, self.out_dim, node, node)
		qk_x_cl = qk_x_cl.expand(batchsize, self.out_dim, node, node)

		total_qk_x = mask_homo_ex * qk_x_cl + mask_hetero_ex * qk_x

		mask = (total_qk_x <= total_base)
		qk_xm = total_qk_x.masked_fill(mask, value=torch.tensor(1e-9))		# masking
		x_a = qk_xm * x
		x_c =  self.conv_1(x) + self.conv_2(x_a)  # (batchsize,feature,node,1)
		x_C = x_c.expand(batchsize, self.out_dim, node, node)
		x_R = x_C.permute(0, 1, 3, 2)
		x = x_C + x_R

		se = torch.mean(x, (2, 3))
		se = self.sed(se)
		se = F.relu(se)
		se = self.seu(se)
		se = torch.sigmoid(se)
		se = se.unsqueeze(2).unsqueeze(3)

		x = x.mul(se)
		x = x + res

		return x, x_c

class GPC_SE_C(nn.Module):

	def __init__(self, in_dim, out_dim):
		super(GPC_SE_C, self).__init__()
		self.out_dim = out_dim
		self.conv = nn.Conv2d(in_dim, out_dim, (1,node))
		nn.init.normal_(self.conv.weight, std=math.sqrt(2*(1-lamda)/(node*in_dim+node*out_dim)))
		self.convres = nn.Conv2d(in_dim, out_dim, 1)
		nn.init.normal_(self.convres.weight, std=math.sqrt(4*lamda/(in_dim+out_dim)))

		self.sed = nn.Linear(out_dim, int(out_dim/SE), bias=False)
		self.seu = nn.Linear(int(out_dim/SE), out_dim, bias=False)

	def forward(self, x):
		batchsize = x.shape[0]

		res = self.convres(x)
		x_c = self.conv(x)
		x_C = x_c.expand(batchsize,self.out_dim,node,node)
		x_R = x_C.permute(0,1,3,2)
		x = x_C+x_R

		se = torch.mean(x,(2,3))
		se = self.sed(se)
		se = F.relu(se)
		se = self.seu(se)
		se = torch.sigmoid(se)
		se = se.unsqueeze(2).unsqueeze(3)

		x = x.mul(se)
		x = x+res

		return x, x_c

class EP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EP, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (1, node))
        nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))

    def forward(self, x):
        x = self.conv(x)
        return x

class NP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NP, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, (node, 1))
        nn.init.normal_(self.conv.weight, std=math.sqrt(4/(node*in_dim+out_dim)))

    def forward(self, x):
        x = self.conv(x)
        return x

class HM_AGPC(nn.Module):
	def __init__(self, GPC_dim_1, GPC_dim_2, GPC_dim_3, EP_dim, NP_dim):
		super(HM_AGPC, self).__init__()

		print('AG_GCN_SE_Masked_Multi_inout_ab')
		self.node_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 4, 4, 0,
						  0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 2, 2, 2, 2, 2, 2,
						  2, 2, 2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5, 5,
						  5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
		# frontal 0 temporal 1 parietal 2 occipital 3  insula 4
		# this is the partition prior for AAL atlas, should be changed based on the chosen atlas.
		self.GPC_1 = GPC_SE_C(1, GPC_dim_1)
		self.GPC_2 = HM_AG_GPC_SE(GPC_dim_1, GPC_dim_2, self.node_list)
		self.GPC_3 = HM_AG_GPC_SE(GPC_dim_2, GPC_dim_3, self.node_list)
		self.EP = EP(GPC_dim_3, EP_dim)
		self.NP = NP(EP_dim, NP_dim)
		self.embedding = nn.Embedding(6, 4).cuda()

		self.fc = nn.Linear(NP_dim, 2)
		nn.init.constant_(self.fc.bias, 0)

	def forward(self, x):
		x, x_c= self.GPC_1(x)
		x = F.relu(x)

		x, x_c= self.GPC_2(x, x_c, self.embedding)
		x = F.relu(x)

		x, x_c= self.GPC_3(x, x_c, self.embedding)
		x = F.relu(x)

		x = self.EP(x)
		x = F.relu(x)

		x = self.NP(x)
		x = F.relu(x)

		x = x.view(x.size(0), -1)

		x = self.fc(x)

		return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        #nn.init.kaiming_normal_(m.weight, mode='fan_out')
		#nn.init.xavier_uniform_(m.weight)
		nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        #nn.init.constant_(m.bias, 0)
