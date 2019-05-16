import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as data_utils
import torch.nn as nn

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

class StructuredSelfAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding including regularization
    and without pruning. Slight modifications have been done for speedup
    """
   
    def __init__(self,batch_size,lstm_hid_dim,d_a,r,max_len,emb_dim=100,vocab_size=None,use_pretrained_embeddings = False,embeddings=None,type=0,n_classes = 1):
        """
        Initializes parameters suggested in paper
 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            max_len     : {int} number of lstm timesteps
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            use_pretrained_embeddings: {bool} use or train your own embeddings
            embeddings  : {torch.FloatTensor} loaded pretrained embeddings
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
 
        Raises:
            Exception
        """
        super(StructuredSelfAttention,self).__init__()
       
        self.embeddings,emb_dim = self._load_embeddings(use_pretrained_embeddings,embeddings,vocab_size,emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,1,batch_first=True)
        self.n_classes = n_classes
        self.batch_size = batch_size       
        self.max_len = max_len
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = self.init_hidden()
        self.r = r
        self.type = type

        self.numBrch = 2
        self.conv = nn.ModuleList([nn.Conv2d(1, 2, (1, 100)),nn.Conv2d(1, 2, (1, 100))])
        self.linear = nn.ModuleList([torch.nn.Linear(2,2),torch.nn.Linear(2,2)])


        # before classification, two branch nets erase each other. 

                 
    def _load_embeddings(self,use_pretrained_embeddings,embeddings,vocab_size,emb_dim):
        """Load the embeddings based on flag"""
       
        if use_pretrained_embeddings is True and embeddings is None:
            raise Exception("Send a pretrained word embedding as an argument")
           
        if not use_pretrained_embeddings and vocab_size is None:
            raise Exception("Vocab size cannot be empty")
   
        if not use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(vocab_size,emb_dim,padding_idx=0)
            
        elif use_pretrained_embeddings:
            word_embeddings = torch.nn.Embedding(embeddings.size(0), embeddings.size(1))
            word_embeddings.weight = torch.nn.Parameter(embeddings)
            emb_dim = embeddings.size(1)
            
        return word_embeddings,emb_dim
       
        
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
 
       
        """
 
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
       
        
    def init_hidden(self):
        if torch.cuda.is_available():
            ret = (Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)).cuda(),Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)).cuda())
        else:
            ret = (Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)),Variable(torch.zeros(1,self.batch_size,self.lstm_hid_dim)))
        return ret

    def getAttention(self,classid):
        # wts = self.linear_final.weight.data[classid.type(device.LongTensor)]
        # att = torch.bmm(wts.unsqueeze(1),self.heatmaps.squeeze()).squeeze() #torch.Size([512, 200])
        att = torch.gather(self.heatmaps[0],1,classid.unsqueeze(1).unsqueeze(1).repeat(1,1,self.heatmaps[0].shape[2])).squeeze()#self.heatmaps[:,classid.type(device.LongTensor)]
        att += torch.gather(self.heatmaps[1],1,classid.unsqueeze(1).unsqueeze(1).repeat(1,1,self.heatmaps[1].shape[2])).squeeze()#self.heatmaps[:,classid.type(device.LongTensor)]
        att = att/2
        return att

    def maskHeatmaps(hm, hm_sub, label): # hm [512,2,200]
        label_rep = label.unsqueeze(1).unsqueeze(1).repeat(1,1,hm.shape[2])
        heatmap = torch.gather(hm,1,label_rep).squeeze() #[512,200]
        att_sub = 1 - F.softmax(torch.gather(hm_sub,1,label_rep).squeeze(),dim=1) #[512,200]
        heatmap_msked = heatmap*att_sub
        ret = Variable(torch.zeros_like(hm)).cuda()
        ret.scatter_(1, label_rep, heatmap_msked)
        ret.scatter_(1, 1-label_rep, hm)
        return ret


    def forward(self,x,brch,label):
        embeddings = self.embeddings(x)       
        outputs, hidden_state = self.lstm(embeddings.view(self.batch_size,self.max_len,-1),self.init_hidden())  

        # branches
        self.heatmaps = []
        for i in range(self.numBrch):
            feats = self.conv[i](outputs.unsqueeze(1)).squeeze().transpose(1,2) #torch.Size([512, 2, 200, 1])
            self.heatmaps.append(self.linear[i](feats).transpose(1,2))

        # brch output
        pred = F.log_softmax(torch.mean(self.heatmaps[brch],dim=2).squeeze())
        # masked brch output
        msk_pred = F.log_softmax(torch.mean(self.maskHeatmaps(self.heatmaps[brch], self.heatmaps[1-brch].detach(), label),dim=2).squeeze())
        # other brch output for adv
        adv_pred = F.log_softmax(torch.mean(self.maskHeatmaps(self.heatmaps[1-brch].detach(), self.heatmaps[brch], label),dim=2).squeeze())

        return pred, msk_pred, adv_pred
       
	   
	#Regularization
    def l2_matrix_norm(self,m):
        """
        Frobenius norm calculation
 
        Args:
           m: {Variable} ||AAT - I||
 
        Returns:
            regularized value
 
       
        """
        return torch.sum(torch.sum(torch.sum(m**2,1),1)**0.5).type(device.DoubleTensor)
