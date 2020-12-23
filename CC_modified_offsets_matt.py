import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import matplotlib.pyplot as plt



def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)

def warpImage(input,field):
    B,C,H,W=field.shape
    #print(B,C,H,W)
    _,_,im_h,im_w = input.shape
    regular_grid=F.affine_grid(torch.eye(2,3).unsqueeze(0).repeat([B,1,1]),(B,C,im_h,im_w)).cuda()
    #print('regular grid shape',regular_grid.shape)
    warped_grid=regular_grid+field.permute([0,2,3,1])
    warped_image=F.grid_sample(input.float(),warped_grid,mode='bilinear')
    return warped_image




class CC_module(nn.Module):
     
    def __init__(self,in_dim, in_height, in_width):
        super(CC_module, self).__init__()
        #self.batchsize = 16                         #New
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        
        #self.gamma = nn.Parameter(torch.zeros(in_dim, in_height, in_width))
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.in_height = in_height
        self.in_width = in_width
        #self.offsets = nn.Parameter(torch.FloatTensor(in_height*in_width).uniform_(0, in_height*in_width))  # New
        self.offsets = nn.Parameter(torch.randn(1,2,in_height+2,in_width+2))  #12 because of avg pool thrice        
        self.deform_weighting = 1   #nn.Parameter(torch.randn(1))
        #self.deform_weighting_y = nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        
        deform = F.avg_pool2d(F.avg_pool2d(F.avg_pool2d(self.offsets,3,stride=1),1,stride=1),1,stride=1).view(2,-1).t()
        #deform = F.avg_pool2d(F.avg_pool2d(F.avg_pool2d(self.offsets,3,stride=1),3,stride=1),3,stride=1).view(2,-1).t()
        #deform = F.avg_pool2d(F.avg_pool2d(F.avg_pool2d(self.offsets,3,stride=1),3,stride=1),3,stride=1).view(2,-1).t()
        
        crisscross = F.affine_grid(torch.eye(2,3).unsqueeze(0),(1,1,self.in_height,self.in_width),align_corners=True).view(-1,2)
        #print('crisscross shape ',crisscross.shape)
        
        #plt.plot(crisscross[:,0],crisscross[:,1],'r+')
        #plt.show()
        deform_x =  deform[(crisscross[:,0]==0),:]
        deform_y =  deform[(crisscross[:,1]==0),:]
        #print('deform_x and y ',deform_x.shape,deform_y.shape,deform.shape)
        
        m_batchsize, channels, height, width = x.size()
        #print('Input Size for CC Module',height,width)
        proj_query = self.query_conv(x)
        
        #print('Proj Query',proj_query[0,0,0:10,0:10])
        #print('Proj Query shape and mean',proj_query.shape,proj_query[0,0,:,:].mean().mean())
        
        ch_q = proj_query.shape[1]
        #bat_q = proj_query.shape[0]
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        
        proj_key = self.key_conv(x)
        ch_k = proj_key.shape[1]
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        #print('Proj key shape ',proj_key_H.shape,proj_key_W.shape)
        
        proj_value = self.value_conv(x)
        ch_v = proj_value.shape[1]
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        #print('Proj Value shape ',proj_value_H.shape,proj_value_W.shape)
        
        #proj_query_tmp = proj_query[0,0,:,:].view(-1)[self.offsets.long()].view(proj_query[0,0,:,:].size())
        #proj_query = proj_query_tmp.repeat(m_batchsize,ch_q,1,1)
        
        #grid_sample() expects the grid with size BxHxWxC
        #displacement field input has size Bx2xHxW
        
        #Regularization of deformation
        regloss = torch.mean((self.offsets[:,0:-1,:]-self.offsets[:,1:,:])**2)+torch.mean((self.offsets[:,:,0:-1]-self.offsets[:,:,1:])**2)
        
        
        #print('Offsets Query shape',self.offsets.shape)
        #warped_query = warpImage(proj_query,self.offsets.unsqueeze(0).repeat(m_batchsize,1,1,1))
        
        #print('Warped Query',warped_query[0,0,0:10,0:10])
        #print('Warped Query shape and mean',warped_query.shape,warped_query[0,0,:,:].mean().mean())

        #off2 = torch.tensor(self.offsets)
        #for bc in range(m_batchsize*channels-1):
        #    off2 = torch.cat((off2,self.offsets+(bc+1)*(height*width)),0)
        
        #proj_query = proj_query.view(-1)[off2.long().view(-1)[0:m_batchsize*ch_q*height*width]].view(proj_query.size()) 
        
                 
        
        k = 0
        
        grid = torch.linspace(-1,1,self.in_height)
        for i in grid :
            if k >= height :
                break
            cc  = crisscross[(crisscross[:,0]== i),:].cuda()
            
            #print('k =',k)
            #print('deform shape and deformx',deform.shape,deform_x.shape)
            #print('size cc and deformx',cc.shape,deform_x.shape)
            
            cc  = cc + (  deform_x)   # Deformation   #self.deform_weighting*
            cc = cc.repeat(m_batchsize,1,1,1)
            img_tmp_Q = F.grid_sample(proj_query,cc.view(m_batchsize,1,-1,2),align_corners=True)
            #print('img_tmp Q size',img_tmp_Q.shape)
            img_tmp_Q = img_tmp_Q.squeeze(2).permute(0,2,1)
            #print('img_tmp size after squeeze ',img_tmp_Q.shape)
            proj_query_H[k,:,:] = img_tmp_Q[0,:,:][1:,:]   #start from 1 as img_tmp is one size larger
            #proj_query_H[k+height,:,:] = img_tmp_Q[1,:,:][1:,:]
            
            img_tmp_K = F.grid_sample(proj_key,cc.view(m_batchsize,1,-1,2),align_corners=True)
            #print('img_tmp_K size',img_tmp_K.shape)
            img_tmp_K = img_tmp_K.squeeze(2)                  #.permute(0,2,1)
            #print('img_tmp_K size',img_tmp_K.shape)
            proj_key_H[k,:,:] = img_tmp_K[0,:,:][:,1:]
            #proj_key_H[k+height,:,:] = img_tmp_K[1,:,:][:,1:]
            
            img_tmp_V = F.grid_sample(proj_value,cc.view(m_batchsize,1,-1,2),align_corners=True)
            #print('img_tmp V size',img_tmp_V.shape)
            img_tmp_V = img_tmp_V.squeeze(2)                 #.permute(0,2,1)
            #print('img_tmp size',img_tmp_V.shape)
            proj_value_H[k,:,:] = img_tmp_V[0,:,:][:,1:]
            #proj_value_H[k+height,:,:] = img_tmp_V[1,:,:][:,1:]
            
            k=k+1
        
        k = 0 
        for i in grid :
            if k >= height :
                break
            cc  = crisscross[(crisscross[:,1]== i),:].cuda()
            
            #print('deformy',deform_y.shape,deform_y.shape)
            cc = cc + ( deform_y)   #self.deform_weighting *
            cc = cc.repeat(m_batchsize,1,1,1)
            img_tmp_Q = F.grid_sample(proj_query,cc.view(m_batchsize,1,-1,2),align_corners=True)
            #print('img_tmp_Q Vert size',img_tmp_Q.shape)
            img_tmp_Q = img_tmp_Q.squeeze(2).permute(0,2,1)
            #print('img_tmp size',img_tmp.shape)
            proj_query_W[k,:,:] = img_tmp_Q[0,:,:][1:,:]
            #proj_query_W[k+height,:,:] = img_tmp_Q[1,:,:][1:,:]
            
            img_tmp_K = F.grid_sample(proj_key,cc.view(m_batchsize,1,-1,2),align_corners=True)
            #print('img_tmp size',img_tmp_K.shape)
            img_tmp_K = img_tmp_K.squeeze(2)               #.permute(0,2,1)
            #print('img_tmp size',img_tmp.shape)
            proj_key_W[k,:,:] = img_tmp_K[0,:,:][:,1:]
            #proj_key_W[k+height,:,:] = img_tmp_K[1,:,:][:,1:]
            
            img_tmp_V = F.grid_sample(proj_value,cc.view(m_batchsize,1,-1,2),align_corners=True)
            #print('img_tmp size',img_tmp_V.shape)
            img_tmp_V = img_tmp_V.squeeze(2)              #.permute(0,2,1)
            #print('img_tmp size',img_tmp.shape)
            proj_value_W[k,:,:] = img_tmp_V[0,:,:][:,1:]
            #proj_value_W[k+height,:,:] = img_tmp_V[1,:,:][:,1:]
            
            
            k=k+1
        
        
        
        

        #for b in range(m_batchsize):
        #    for c in range(ch_k) :
        #        proj_key[b,c,:,:] = proj_key[b,c,:,:].view(-1)[self.offsets.long()].view(proj_key[b,c,:,:].size())              #New
        
        
      
        
        
        #proj_key = proj_key.view(-1)[off2.long().view(-1)[0:m_batchsize*ch_k*height*width]].view(proj_key.size()) 
        #proj_key = proj_key.view(-1)[self.offsets.repeat(m_batchsize,ch_k,1,1).long()].view(proj_key.size()) 
                
        

        
        
        #for b in range(m_batchsize):
        #    for c in range(ch_v) :
        #        proj_value[b,c,:,:] = proj_value[b,c,:,:].view(-1)[self.offsets.long()].view(proj_value[b,c,:,:].size())              #New
        
        #proj_value_tmp = proj_value[0,0,:,:].view(-1)[self.offsets.long()].view(proj_value[0,0,:,:].size())
        #proj_value = proj_value_tmp.repeat(m_batchsize,ch_v,1,1)
        
       
        
        #proj_value = proj_value.view(-1)[off2.long().view(-1)[0:m_batchsize*ch_v*height*width]].view(proj_value.size()) 
        #proj_value = proj_value.view(-1)[self.offsets.repeat(m_batchsize,ch_v,1,1).long()].view(proj_value.size()) 
        
        
       
        
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        out_HW = out_H + out_W
        return self.gamma*(out_H + out_W) + x , regloss  #return torch.mul(self.gamma, out_HW) + x



if __name__ == '__main__':
    model = CC_module(2048)
    x = torch.randn(4, 2048, 41, 41)
    out = model(x)
    print(out.shape)
