import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

"""Parameter是Tensor，即 Tensor 拥有的属性它都有，⽐如可以根据data 来访问参数数值
nn.Parameters 与 register_parameter 都会向 _parameters写入参数，但是后者可以支持字符串命名。
从源码中可以看到，nn.Parameters为Module添加属性的方式也是通过register_parameter向 _parameters写入参数。
register_parameter(name, param)向我们建立的网络module添加 parameter
"""
class GraphConvolution(nn.Module):
    """
       Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
       """
    def __init__(self,in_features,out_features,bias=True):
        super(GraphConvolution,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=Parameter(torch.FloatTensor(in_features,out_features))
        if bias:
            self.bias=Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias',None)
        def reset_parameters(self):
            stdv=1./math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv,stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv,stdv)
        """
        只有自己定义的参数，例如weight与bias才需要自定义初始化。一般在__init__层里，调用self.reset_parameters()来实现
                在self.reset_parameters()函数里，使用两种方法:
                第一种：nn.init.xavier_uniform_(x, gain=nn.init.calculate_gain(‘relu’))。其中， gain 参数来自定义初始化的标准差
            来匹配特定的激活函数：
            -第二种：变量.data.uniform_(-stdv, stdv)
        """
        def forward(self,input,adj):
            support=torch.mm(input,self.weight)
            output=torch.spmm(adj,support)
            if self.bias is not None:
                return  self.bias+output
            else:
                return output

        def __repr__(self):
            return self.__class__.__name__ + ' (' \
                   + str(self.in_features) + ' -> ' \
                   + str(self.out_features) + ')'




