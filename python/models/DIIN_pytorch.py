import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import collections
from util import blocks_torch
from my.tensorflow.general_torch import flatten, reconstruct, exp_mask
import math
#from memory_profiler import profile

class DIIN(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, config, seq_length, emb_dim, hidden_dim, emb_train, embeddings = None, pred_size = 3, context_seq_len = None, query_seq_len = None, dropout_rate = 0.0):
        super(DIIN, self).__init__()

        self.embedding_dim = emb_dim
        self.dim = hidden_dim
        self.sequence_length = seq_length
        self.pred_size = pred_size 
        self.context_seq_len = context_seq_len
        self.query_seq_len = query_seq_len
        self.dropout_rate = dropout_rate
        self.config = config
        self.dropout = nn.Dropout(p=0.0)

        self.char_emb_cnn = nn.Conv2d(8, 100, (1, 5), stride=(1, 1, 1, 1), padding=0, bias=True)
        self.interaction_cnn = nn.Conv2d(448, int(448 * config.dense_net_first_scale_down_ratio), config.first_scale_down_kernel, padding=0)

        self.highway_network_linear = nn.Linear(448, 448, bias=True)
        self.self_attention_linear = nn.Linear(1344, 1, bias=True)
        self.fuse_gate_linear = nn.Linear(31360, 448, bias=True)
        self.final_linear = nn.Linear(5616, 3, bias=True)
        self.test_linear = nn.Linear(308736, 3, bias=True)

        if embeddings is not None:
            #print(embeddings.shape)
            self.emb = nn.Embedding(embeddings.shape[0], embeddings.shape[1], padding_idx=0)
            self.emb.weight.data.copy_(torch.from_numpy(embeddings).type('torch.LongTensor'))
            self.emb.weight.requires_grad = True

        self.char_emb_init = nn.Embedding(config.char_vocab_size, config.char_emb_size)
        self.char_emb_init.weight.requires_grad = False

        self.dense_net = DenseNet(134, config.dense_net_growth_rate, config.dense_net_transition_rate, config.dense_net_layers, config.dense_net_kernel_size)

    def dropout_rate_decay(self, global_step, decay_rate=0.997):
        p = 1 - 1 * 0.997 ** (global_step / 10000)
        self.dropout_rate = p

    #@profile
    def forward(self, premise_x, hypothesis_x, \
                pre_pos, hyp_pos, premise_char_vectors, hypothesis_char_vectors, \
                premise_exact_match, hypothesis_exact_match):
        prem_seq_lengths, prem_mask = blocks_torch.length(premise_x)  # mask [N, L , 1]
        hyp_seq_lengths, hyp_mask = blocks_torch.length(hypothesis_x)    	

        #print(premise_x.size())
        #print(type(premise_x))
        #print(self.dropout_rate)
        #print(self.training)
        #print(premise_x.size(),hypothesis_x.size())
        premise_in = F.dropout(self.emb(premise_x.type('torch.LongTensor')), p = self.dropout_rate,  training=self.training)
        #print(premise_in.size())
        hypothesis_in = F.dropout(self.emb(hypothesis_x.type('torch.LongTensor')), p = self.dropout_rate,  training=self.training)
        #print("premise_in", type(premise_in))
        #print("premise_char_vectors size", premise_char_vectors.size()) #[70, 48, 16]
        conv_pre, conv_hyp = self.char_emb(premise_char_vectors, hypothesis_char_vectors)
        #print(premise_in.size(), conv_pre.size())
        #print("conv_pre", type(conv_pre))
        premise_in = torch.cat([premise_in, conv_pre], 2) #[70, 48, 300], [70, 48, 100] --> [70,48,400]
        hypothesis_in = torch.cat([hypothesis_in, conv_hyp], 2)

        pre_pos = pre_pos.type('torch.FloatTensor')
        #print("pre_pos size:", pre_pos.size())
        #print("pre_pos", pre_pos)
        premise_in = torch.cat([premise_in, pre_pos], 2) # 70*48*447
        #print(premise_in.size()) 
        hyp_pos = hyp_pos.type('torch.FloatTensor')
        hypothesis_in = torch.cat([hypothesis_in, hyp_pos], 2)

        premise_exact_match = torch.unsqueeze(premise_exact_match.type('torch.FloatTensor'),2) #70*48*1
        #print("premise_exact_match size:", premise_exact_match.size()) 
        #print("premise_exact_match: ", premise_exact_match)
        premise_in = torch.cat([premise_in, premise_exact_match], 2) #70*48*448
        #print(premise_in.size())
        hypothesis_exact_match = torch.unsqueeze(hypothesis_exact_match.type('torch.FloatTensor'),2)
        hypothesis_in = torch.cat([hypothesis_in, hypothesis_exact_match], 2) #70*48*448
        

        premise_in = highway_network(self.highway_network_linear, premise_in, self.config.highway_num_layers, True, wd=self.config.wd, is_train = self.training)    
        hypothesis_in = highway_network(self.highway_network_linear, hypothesis_in, self.config.highway_num_layers, True, wd=self.config.wd, is_train = self.training)
        #print("highway_network premise_in", type(premise_in))
        #print('pre_in',premise_in.size())
        pre = premise_in  #[70, 48, 448]
        hyp = hypothesis_in
        #import pdb
        
        for i in range(self.config.self_att_enc_layers):
            pre = self_attention_layer(self.self_attention_linear, self.fuse_gate_linear, self.config, self.training, pre, input_drop_prob=self.dropout_rate, p_mask=prem_mask) # [N, len, dim]    
            hyp = self_attention_layer(self.self_attention_linear, self.fuse_gate_linear, self.config, self.training, hyp, input_drop_prob=self.dropout_rate, p_mask=prem_mask)
            #pre = p
            #hyp = h
        #print("self_attention pre", type(pre))
        #print('pre:',pre.size())  #[70, 48, 448]
        """
        def model_one_side(config, main, support, main_length, support_length, main_mask, support_mask):
            bi_att_mx = bi_attention_mx(config, self.training, main, support, p_mask=main_mask, h_mask=support_mask) # [N, PL, HL]
            print("bi_attention_mx", type(bi_att_mx))
            bi_att_mx = F.dropout(bi_att_mx, p=self.dropout_rate, training=self.training)
            out_final = dense_net(config, bi_att_mx, self.training)
            print("dense_net", type(out_final))
            return out_final

        premise_final = model_one_side(self.config, pre, hyp, prem_seq_lengths, hyp_seq_lengths, prem_mask, hyp_mask)
        """
        #pdb.set_trace()
        bi_att_mx = bi_attention_mx(self.config, self.training, pre, hyp, p_mask=prem_mask, h_mask=hyp_mask) # [N, PL, HL]
        print("bi_attention_mx", bi_att_mx.size()) # 70,448,48,48
        bi_att_mx = F.dropout(bi_att_mx, p=self.dropout_rate, training=self.training)

        fm = self.interaction_cnn(bi_att_mx)
        print('fm:',fm.size()) # [70, 134, 48, 48]
        if self.config.first_scale_down_layer_relu:
            fm = F.relu(fm)

        #premise_final = dense_net(self.config, fm, self.training)
        #print("dense_net", type(premise_final))

        #print("dense_net", fm.size())
        #premise_final = self.dense_net(fm)
        fm = fm.view(70, -1)
        logits = self.test_linear(fm)
        #print("premise_final", premise_final.size())
        #premise_final = premise_final.view(self.config.batch_size, -1)
        #print("premise_final", premise_final.size())

        #logits = linear(self.final_linear, [premise_final], self.pred_size ,True, bias_start=0.0, squeeze=False, wd=self.config.wd, input_drop_prob=self.config.keep_rate,
                                #is_train=self.training)

        return logits

    def char_emb(self, premise_char, hypothesis_char):
        input_shape = premise_char.size()
        bs = premise_char.size(0)
        seq_len = premise_char.size(1)
        word_len = premise_char.size(2)

        premise_char = premise_char.view(-1, word_len) # (N*seq_len, word_len)
        char_pre = self.char_emb_init(premise_char)#.type('torch.LongTensor')) # (N*seq_len, word_len, embd_size)
        #char_pre = char_pre.view(*input_shape, -1) # (N, seq_len, word_len, embd_size)
        char_pre = char_pre.view(bs, -1, seq_len, word_len)

        hypothesis_char = hypothesis_char.view(-1, word_len)
        char_hyp = self.char_emb_init(hypothesis_char)#.type('torch.LongTensor'))
        char_hyp = char_hyp.view(bs, -1, seq_len, word_len)

        #char_hyp = char_hyp.view(*input_shape, -1) # (N, seq_len, word_len, embd_size)

        filter_sizes = list(map(int, self.config.out_channel_dims.split(','))) #[100]
        #print('filter:',filter_sizes)
        heights = list(map(int, self.config.filter_heights.split(',')))
        #print('heights:',heights)        #[5]
        assert sum(filter_sizes) == self.config.char_out_size, (filter_sizes, self.config.char_out_size)    	

        def multi_conv1d(char_pre, filter_sizes, heights):
            assert len(filter_sizes) == len(heights)
            outs = []
            for filter_size, height in zip(filter_sizes, heights):
                if filter_size == 0:
                    continue
                char_pre = F.dropout2d(char_pre, p=self.dropout_rate, training=self.training) #[70, 48, 16, 8] --
                #print('char_pre:', char_pre.size())
                #cnn2d = nn.Conv2d(char_pre.size()[-1], filter_size, (1, height), stride=(1, 1, 1, 1), padding=0, bias=True)
                cnn_pre = self.char_emb_cnn(char_pre) #[70, 100, 48, 12]
                #print('cnn_pre:',cnn_pre.size())  
                out = torch.max(F.relu(cnn_pre), 3)[0]  #[70, 100, 48]
                #print('out:',out.size()) 
                outs.append(out)

            concat_out = torch.cat(outs, 2) #[70, 100, 48]
            #print('concat_out:',concat_out.size()) 
            return concat_out


        conv_pre = multi_conv1d(char_pre, filter_sizes, heights) # [70*100*48]
        conv_hyp = multi_conv1d(char_hyp, filter_sizes, heights) # [70*100*48]
        #print(type(conv_pre))
        #print(self.sequence_length, self.config.char_out_size)
        conv_pre = conv_pre.view(-1, self.sequence_length, self.config.char_out_size)
        conv_hyp = conv_hyp.view(-1, self.sequence_length, self.config.char_out_size) # [70*48*100]
        #print('conv_pre:',conv_pre.size())
        return conv_pre, conv_hyp
   	
def linear(linear_layer, data_in, output_size, bias, bias_start=0.0, squeeze=False, wd=0.0, input_drop_prob=0, is_train = None):
    flat_datas = [flatten(data, 1) for data in data_in]
    #print('flat_datas,',len(flat_datas),flat_datas[0].size())
    assert is_train is not None
    flat_datas = [F.dropout(data, p=input_drop_prob, training=is_train) for data in flat_datas]
    #print(len(flat_datas))

    total_data_size = sum([data.size()[1] for data in flat_datas])
    #_linear = nn.Linear(total_data_size, output_size, bias=bias)
    #print(total_data_size, output_size)
    if len(flat_datas) > 1:
        flat_datas = torch.cat(flat_datas, 1)
    else:
        flat_datas = flat_datas[0]
    flat_out = linear_layer(flat_datas)
    #print("flat_out size, ", flat_out.size())
    out = reconstruct(flat_out, data_in[0], 1)
    #print("out size, ", out.size())
    if squeeze:
        #print(len(list(data_in[0].size()))-1) #3
        out = torch.squeeze(out, len(list(data_in[0].size()))-1)
    # if wd:add_wd(wd)
    return out

def highway_network(highway_network_linear, data_in, num_layers, bias, bias_start=0.0, wd=0.0, input_drop_prob=0, is_train=None, output_size = None):  		
    def highway_layer(highway_network_linear, data_in, bias, bias_start=0.0, wd=0.0, input_drop_prob=0, is_train = None, output_size = None):
        if output_size is not None:
            d = output_size
        else:
            d = data_in.size()[-1]
        trans = linear(highway_network_linear, [data_in], d, bias, bias_start=bias_start, wd=wd, input_drop_prob=input_drop_prob, is_train = is_train)
        trans = F.relu(trans)
        gate = linear(highway_network_linear, [data_in], d, bias, bias_start=bias_start, wd=wd, input_drop_prob=input_drop_prob, is_train = is_train)
        gate = F.sigmoid(gate)
        if d != data_in.size()[-1]:
            data_in = linear(highway_network_linear, [data_in], d, bias, bias_start=bias_start, wd=wd, input_drop_prob=input_drop_prob, is_train = is_train)
        out = gate * trans + (1 - gate) * data_in
        return out

    prev = data_in
    for layer_idx in range(num_layers):
        cur = highway_layer(highway_network_linear, prev, bias, bias_start=bias_start, wd=wd, 
            input_drop_prob=input_drop_prob, is_train=is_train, output_size = output_size)
        prev = cur
    return cur



def self_attention(linear_layer, config, is_train, p, p_mask=None): #[N, L, 2d]
    PL = p.size()[1]
    dim = p.size()[-1]
    p_aug_1 = torch.unsqueeze(p, 2).repeat(1,1,PL,1)
    p_aug_2 = torch.unsqueeze(p, 1).repeat(1, PL, 1, 1)

    if p_mask is None:
        ph_mask = None
    else:
        p_mask_aug_1 = torch.unsqueeze(p_mask, 2).repeat(1, 1, PL, 1).data.numpy().any(axis=3)
        p_mask_aug_2 = torch.unsqueeze(p_mask, 1).repeat(1, PL, 1, 1).data.numpy().any(axis=3)
        self_mask = Variable(torch.from_numpy((p_mask_aug_1 & p_mask_aug_2).astype(int)).type('torch.IntTensor'), requires_grad=False)
    #print(self_mask)

    h_logits = get_logits(linear_layer, [p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                          is_train=is_train, func=config.self_att_logit_func)  # [N, PL, HL]
    self_att = softsel(p_aug_2, h_logits) 
    return self_att

def self_attention_layer(self_attention_layer, fuse_gate_linear, config, is_train, p, input_drop_prob, p_mask=None):
    PL = p.size()[1]
    self_att = self_attention(self_attention_layer, config, is_train, p, p_mask=p_mask)

    #print("self_att shape")
    #print(self_att.size())  # [70, 48, 448]

    p0 = fuse_gate(fuse_gate_linear, config, is_train, p, self_att, input_drop_prob)
    
    return p0

def fuse_gate(fuse_gate_linear, config, is_train, lhs, rhs, input_drop_prob):
    dim = list(lhs.size())[-1]
    lhs_1 = linear(fuse_gate_linear, lhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
    rhs_1 = linear(fuse_gate_linear, rhs, dim ,True, bias_start=0.0, squeeze=False, wd=0.0, input_drop_prob=input_drop_prob, is_train=is_train)
    if config.self_att_fuse_gate_residual_conn and config.self_att_fuse_gate_relu_z:
        z = F.relu(lhs_1 + rhs_1)
    else:
        z = F.tanh(lhs_1 + rhs_1)
    lhs_2 = linear(fuse_gate_linear, lhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
    rhs_2 = linear(fuse_gate_linear, rhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
    f = F.sigmoid(lhs_2 + rhs_2)
    print(config.two_gate_fuse_gate)
    if config.two_gate_fuse_gate:
        lhs_3 = linear(fuse_gate_linear, lhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
        rhs_3 = linear(fuse_gate_linear, rhs, dim ,True, bias_start=0.0, squeeze=False, wd=config.wd, input_drop_prob=input_drop_prob, is_train=is_train)
        f2 = F.sigmoid(lhs_3 + rhs_3)
        out = f * lhs + f2 * z
    else:   
        out = f * lhs + (1 - f) * z

    return out


def double_linear_logits(args, size, bias, bias_start=0.0, mask=None, wd=0.0, input_drop_prob=0.0, is_train=None):
	first = torch.tanh(linear(args, size, bias, bias_start=bias_start,
		wd=wd, input_drop_prob=input_drop_prob, is_train=is_train))
	second = linear(first, 1, bias, bias_start=bias_start, squeeze=True,
		wd=wd, input_drop_prob=input_drop_prob, is_train=is_train)
	if mask is not None:
		second = exp_mask(second, mask)
	return second


def linear_logits(linear_layer, args, bias, bias_start=0.0, mask=None, wd=0.0, input_drop_prob=0.0, is_train=None):
	logits = linear(linear_layer, args, 1, bias, bias_start=bias_start, squeeze=True,
		wd=wd, input_drop_prob=input_drop_prob, is_train=is_train)
	if mask is not None:
		logits = exp_mask(logits, mask)
	return logits


def sum_logits(args, mask=None):
    rank = len(args[0].size())
    logits = sum(torch.sum(arg, rank-1) for arg in args)
    if mask is not None:
        logits = exp_mask(logits, mask)
    return logits


def get_logits(linear_layer, args, size, bias, bias_start=0.0, mask=None, wd=0.0, input_drop_prob=0.0, is_train=None, func=None):
    if func is None:
        func = "sum"
    if func == 'sum':
        return sum_logits(args, mask=mask)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, mask=mask, wd=wd, input_drop_prob=input_drop_prob,
                             is_train=is_train)
    elif func == 'double':
        return double_linear_logits(args, size, bias, bias_start=bias_start, mask=mask, wd=wd, input_drop_prob=input_drop_prob,
                                    is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask)
    elif func == 'scaled_dot':
        assert len(args) == 2
        dim = args[0].get_shape().as_list()[-1]
        arg = args[0] * args[1]
        arg = arg / tf.sqrt(tf.constant(dim, dtype=tf.float32))
        return sum_logits([arg], mask=mask)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, mask=mask, wd=wd, input_drop_prob=input_drop_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, wd=wd, input_drop_prob=input_drop_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits(linear_layer, [args[0], args[1], new_arg], bias, bias_start=bias_start, mask=mask, wd=wd, input_drop_prob=input_drop_prob,
                             is_train=is_train)
    else:
        raise Exception()

def softsel(target, logits, mask=None):
    a = softmax(logits, mask=mask)
    target_rank = len(target.size())
    out = torch.sum(torch.unsqueeze(a, -1) * target, target_rank - 2)
    return out

def softmax(logits, mask=None):
    if mask is not None:
        logits = exp_mask(logits, mask)
    flat_logits = flatten(logits, 1)
    flat_out = F.softmax(flat_logits)
    out = reconstruct(flat_out, logits, 1)

    return out

def bi_attention_mx(config, is_train, p, h, p_mask=None, h_mask=None): #[N, L, 2d]
    PL = p.size()[1] 
    HL = h.size()[1]
    p_aug = torch.unsqueeze(p, 2).repeat(1,1,HL,1)
    h_aug = torch.unsqueeze(h, 1).repeat(1,PL,1,1) #[N, PL, HL, 2d]

    """
    if p_mask is None:
        ph_mask = None
    else:
        p_mask_aug = torch.unsqueeze(p_mask, 2).repeat(1, 1, HL, 1).data.numpy().any(axis=3)
        h_mask_aug = torch.unsqueeze(h_mask, 1).repeat(1, PL, 1, 1).data.numpy().any(axis=3)
        ph_mask = Variable(torch.from_numpy((p_mask_aug & h_mask_aug).astype(int)).type('torch.FloatTensor'))

    ph_mask = None##########################??????????????????????
    """
    h_logits = p_aug * h_aug
    #print('bi_attention_out',h_logits.size()) [70,48,48,448]
    h_logits = h_logits.view(config.batch_size, -1, PL, HL)
    return h_logits 

class Dense_net_block(nn.Module):
    def __init__(self, outChannels, growth_rate, kernel_size):
        super(Dense_net_block, self).__init__()
        self.conv = nn.Conv2d(outChannels, growth_rate, kernel_size=kernel_size, bias=False, padding=1)
        #print('block',outChannels, growth_rate)

    def forward(self, x):
        ft = F.relu(self.conv(x))
        #print("ft", ft.size())
        out = torch.cat((x, ft), dim=1)
        return out

class Dense_net_transition(nn.Module):
    def __init__(self, nChannels, outChannels):
        super(Dense_net_transition, self).__init__()
        self.conv = nn.Conv2d(nChannels, outChannels, kernel_size=1, bias=False)
        #print('trans',nChannels, outChannels)

    def forward(self, x):
        out = self.conv(x)
        out = F.max_pool2d(out, (2,2), (2,2), padding=0)
        #print('trans',out.size())
        return out

class DenseNet(nn.Module):
    def __init__(self, nChannels, growthRate, reduction, nDenseBlocks, kernel_size):
        super(DenseNet, self).__init__()
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Dense_net_transition(nChannels, nOutChannels)
        nChannels = nOutChannels
        
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Dense_net_transition(nChannels, nOutChannels)
        nChannels = nOutChannels
       
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, kernel_size)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = Dense_net_transition(nChannels, nOutChannels)


        #self.final_linear = nn.Linear(5616, 3, bias=True)

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, kernel_size):
        layers = []
        for i in range(int(nDenseBlocks)):
            #print(nChannels, growthRate)
            layers.append(Dense_net_block(nChannels, growthRate, kernel_size))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.trans1(self.dense1(x))
        #print("out, ", out.size())
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        #out = out.view(70, -1)
        #out = self.final_linear(out)
        return out


"""
def dense_net(config, fm, is_train):
    #dim = denseAttention.size()[-1]
    #print('dense_net_conv:', dim * config.dense_net_first_scale_down_ratio, config.first_scale_down_kernel)
    #cnn2d = nn.Conv2d(denseAttention.size()[-1], int(dim * config.dense_net_first_scale_down_ratio), config.first_scale_down_kernel, padding=0)
    #print("CNN: ", denseAttention.size()[-1], int(dim * config.dense_net_first_scale_down_ratio), config.first_scale_down_kernel)
    #fm = cnn2d(denseAttention.permute(0,3,1,2)).permute(0,2,3,1) 
    #print('fm:',fm.size()) # [70, 48, 48, 134]
    #if config.first_scale_down_layer_relu:
    #    fm = F.relu(fm)

    fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train) 
    fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate)
    fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train) 
    fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate)
    fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers, config.dense_net_kernel_size, is_train) 
    fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate)

    shape_list = fm.size()
    #print('shape_list:',shape_list)
    fm = fm.contiguous().view(shape_list[0], shape_list[1]*shape_list[2]*shape_list[3])
    #print(fm)
    return fm

def dense_net_block(config, feature_map, growth_rate, layers, kernel_size, is_train, padding=0):
    dim = feature_map.size()[-1]
    #print(growth_rate, kernel_size) [20,3]
    #print('fm:',feature_map.size()) #70x48x48x134
    list_of_features = [feature_map]
    features = feature_map

    for i in range(layers):
        cnn2d = nn.Conv2d(features.size()[-1], growth_rate, (kernel_size, kernel_size), padding=1)
        #print('fm_pre:',features.size())
        print("dense_net_block", features.size()[-1], growth_rate, (kernel_size, kernel_size))
        fm = cnn2d(features.permute(0,3,1,2)).permute(0,2,3,1)
        #print('fm_cnn:',fm.size())
        ft = F.relu(fm)
        #print('ft:',ft.size()) #[70, 48, 48, 20]
        list_of_features.append(ft)  
        #print(len(list_of_features))
        features = torch.cat(list_of_features, dim=3)

    #print("dense net block out shape") # [70,48,48,294]
    #print(list(features.size()))
    return features 

def dense_net_transition_layer(config, feature_map, transition_rate):
    out_dim = int(feature_map.size()[-1] * transition_rate)
    #print('out_dim', out_dim) # 147
    cnn2d = nn.Conv2d(feature_map.size()[-1], out_dim, 1, padding=0)
    print("dense_net_transition_layer", feature_map.size()[-1], out_dim)
    feature_map = cnn2d(feature_map.permute(0,3,1,2))
    #print('fm_dntl:', feature_map.size()) # [70, 147, 48, 48]
    max_pool = nn.MaxPool2d((2,2), (2,2), padding=0)
    feature_map = max_pool(feature_map).permute(0,2,3,1) 
    
    #print("Transition Layer out shape")
    #print(list(feature_map.size())) # [70, 24，24，147]
    return feature_map
"""

