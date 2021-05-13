import torch.nn as nn

class IDCNN(nn.Module):
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}]
        net = nn.Sequential()

        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module("layer%d"%i, single_block)
            net.add_module("relu", nn.ReLU())
        
        self.linear = nn.Linear(input_size, filters)
        # self.conv_init = nn.Conv1d(
        #     in_channels=input_size,
        #     out_channels=filters,
        #     kernel_size=kernel_size,
        #     padding= (kernel_size -1) // 2
        # )
        self.idcnn = nn.Sequential()


        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())

    def forward(self, embeddings):
        # print('input', embeddings.shape)
        embeddings = self.linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        # embeddings = self.conv_init(embeddings)
        # print('after liner', embeddings.shape)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        # print('output',output.shape)
        return output





