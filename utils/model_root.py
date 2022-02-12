
import torch
import torch.nn as nn



class SimpleConv(nn.Module):
    
  def __init__(self, input_features=1):
    super(SimpleConv, self).__init__()
    self.conv_initial = nn.Conv2d(input_features, 64, 7, 2)
    self.conv_initial1 = nn.Conv2d(64, 256, 3, 2)
    self.max_pool = nn.MaxPool2d(2, 2)
    self.relu = nn.ReLU()
    self.bn1 = nn.BatchNorm2d(64)
    self.bn2 = nn.BatchNorm2d(256)
    self.bn3 = nn.BatchNorm2d(512)
    self.conv1_incp = nn.Conv2d(256, 256, 1, padding='same')
    self.conv2_incp = nn.Conv2d(256, 256, 3, padding='same')
    self.conv3_incp = nn.Conv2d(256, 256, 5, padding='same')
    self.max_pool_incp = nn.MaxPool2d(3, 1)
    self.conv_final = nn.Conv2d(1024, 512, 5, padding='same')
    self.flat = nn.Flatten()
    self.linear = nn.Linear(512*7*7, 1024)
    self.linear1 = nn.Linear(1024, 10)
    self.s_max = nn.Softmax(dim=1)
    self.pad_initial = nn.ConstantPad2d((2,3,2,3),0.0)
    self.pad_initial1 = nn.ConstantPad2d((0,1,0,1), 0.0)
    self.pad_incp = nn.ConstantPad2d((1,1,1,1),0.0)
    self.drop = nn.Dropout(p=0.4)
    self.drop_initial = nn.Dropout(p=0.3)


  def forward(self,input):
    
    #_layer_1
    
    output1_initial = self.relu(self.bn1(self.conv_initial(self.pad_initial(input))))
    output2_initial = self.max_pool(output1_initial)
    

    #_layer_2

    output3_initial = self.relu(self.bn2(self.conv_initial1(self.pad_initial1(output2_initial))))
    output3 = self.drop_initial(self.max_pool(output3_initial))


    #_inception_layer
    
    #_inception_1
    output1_incp = self.relu(self.conv1_incp(output3))
    output1_incp = self.relu(self.conv2_incp(output1_incp))
    #_inception_2
    output2_incp = self.relu(self.conv1_incp(output3)) 
    output2_incp = self.relu(self.conv3_incp(output2_incp))
    #_inception_3  
    output3_incp = self.relu(self.conv1_incp(self.max_pool_incp(self.pad_incp(output3))))
    #_inception_4
    output4_incp = self.relu(self.conv1_incp(output3)) 
    #_stack
    output_incp = torch.cat([output1_incp,output2_incp,output3_incp,output4_incp],dim=1)
    output_pool = self.max_pool(output_incp)
    output_pool = self.drop(output_pool)
    

    #_layer_conv_last

    output_final_conv = self.drop(self.relu(self.bn3(self.conv_final(output_pool))))
    output_final = self.s_max(self.linear1(self.linear(self.flat(output_final_conv))))


    return output_final


