import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


######################################
# HIPT w/o Transformers #
######################################
class HIPT_None_FC(nn.Module):
    def __init__(self, in_features=1280, size_arg = "small", dropout=0.25, out_features=2):
        super(HIPT_None_FC, self).__init__()
        self.size_dict_path = {"small": [in_features, 256, 256], "big": [in_features, 512, 384]}
        size = self.size_dict_path[size_arg]

        ### Local Aggregation
        self.local_phi = nn.Sequential(
            nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(0.25),
        )
        self.local_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
        
        ### Global Aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25),
        )
        self.global_attn_pool = Attn_Net_Gated(L=size[1], D=size[1], dropout=0.25, n_classes=1)
        self.global_rho = nn.Sequential(*[nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)])
        self.classifier = nn.Linear(size[1], out_features)
        self.out_features = out_features


    def forward(self, h, **kwargs):
        if h.dim() == 4:
            h = h.squeeze(0) # removing batch dim to match HIPT expected input (dim == 3)
        x_256 = h

        ### Local
        h_256 = self.local_phi(x_256)
        A_256, h_256 = self.local_attn_pool(h_256)  
        A_256 = A_256.squeeze(dim=2) # A = torch.transpose(A, 1, 0)
        A_256 = F.softmax(A_256, dim=1) 
        h_4096 = torch.bmm(A_256.unsqueeze(dim=1), h_256).squeeze(dim=1)
        
        ### Global
        h_4096 = self.global_phi(h_4096)
        A_4096, h_4096 = self.global_attn_pool(h_4096)  
        A_4096 = torch.transpose(A_4096, 1, 0)
        A_4096 = F.softmax(A_4096, dim=1) 
        h_path = torch.mm(A_4096, h_4096)
        h_path = self.global_rho(h_path)
        logits = self.classifier(h_path)

        return {"logits": logits}