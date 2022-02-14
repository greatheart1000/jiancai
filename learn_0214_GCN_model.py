import torch.nn as nn
from learn_0214gcn import GraphConvolution
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self,n_feat,hidden_feat,nclass,dropout):
        super(GCN, self).__init__()
        self.gc1=GraphConvolution(in_features=n_feat,out_features=hidden_feat)
        self.gc2=GraphConvolution(in_features=hidden_feat,out_features=nclass)
        self.dropout=dropout
    def forward(self,x,adj):
        x=F.relu(self.gc1(x,adj))
        x=F.dropout(x,self.dropout,training=self.training)
        x=self.gc2(x,adj)
        return F.log_softmax(x,dim=1)
    """ log_softmax能够解决函数overflow和underflow，加快运算速度，提高数据稳定性
    overflow淹没,漫过,使泛滥 满得溢出,外流,挤出,溢出,漫出,泛滥
    underflow 下漏，下溢; 潜（底）流，地下水流
    数学上log_Softmax是对Softmax取对数
    比如上图中，z1、z2、z3取值很大的时候，超出了float能表示的范围。  同理当输入为负数且绝对值也很大的时候，
    会分子、分母会变得极小，有可能四舍五入为0，导致下溢出。
    """
    """Softmax会存在上溢出和下溢出的情况，这是因为Softmax会进行指数操作，当上一层的输出，
    也就是Softmax的输入比较大的时候，可能会产生上溢出，超出float的能表示范围；同理，当输入为负值
    且绝对值比较大的时候，分子分母会极小，接近0，从而导致下溢出。这时候log_Softmax能够很好的解决溢出问题，
    且可以加快运算速度，提升数据稳定性
     使用log_softmax。 一方面是为了解决溢出的问题，第二个是方便CrossEntropyLoss的计算。不需要担心值域的变化
    """