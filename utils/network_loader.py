import torch

def load_network(network_type, weight_path):
    network = None

    if network_type == 'Restormer':
        from models.Restormer.restormer import Restormer
        network = Restormer()

    elif network_type == 'NAFNet':
        from models.NAFNet.NAFNet_arch import NAFNetLocal

        network = NAFNetLocal(
            width=64,
            enc_blk_nums=[1, 1, 1, 28],
            middle_blk_num=1,
            dec_blk_nums=[1, 1, 1, 1]
        )
        
        torch.use_deterministic_algorithms(True, warn_only=True)

    elif network_type == 'HWMNet':
        from models.HWMNet.HWMNet import HWMNet
        network = HWMNet()

    elif network_type == 'LLFormer':
        from models.LLFormer.LLFormer import LLFormer
        network = LLFormer()
        
    elif network_type == 'LLFormer_Official':
        from models.LLFormer.LLFormer import LLFormer
        network = LLFormer(inp_channels=3,out_channels=3,dim = 16,num_blocks = [2,4,8,16],num_refinement_blocks = 2,heads = [1,2,4,8],ffn_expansion_factor = 2.66,bias = False,LayerNorm_type = 'WithBias',attention=True,skip = False)
        
        network.load_state_dict(torch.load(weight_path)['state_dict'])

        return network

    elif network_type == 'MIRNetv2':
        from models.MIRNetv2.mirnet_v2_arch import MIRNet_v2
        network = MIRNet_v2()

    elif network_type == 'RetinexFormer':
        from models.RetinexFormer.RetinexFormer_arch import RetinexFormer
        network = RetinexFormer(
            in_channels=3,
            out_channels=3,
            n_feat=40,
            stage=1,
            num_blocks=[1, 2, 2]
        )

    elif network_type == 'SNRAware':
        from models.SNRAware.low_light_transformer import low_light_transformer
        network = low_light_transformer(
            nf=64,
            nframes=5,
            groups=8,
            front_RBs=1,
            back_RBs=1,
            predeblur=True,
            HR_in=True,
            w_TSA=True
        )

    elif network_type == 'Uformer':
        from models.Uformer.model import Uformer
        network = Uformer(img_size=256)

    elif network_type == 'UNet':
        from models.unet.unet_model import UNet
        network = UNet(
            n_channels=3,
            n_classes=3
        )
        
    elif network_type == 'DRBN':
        from models.DRBN.drbn import DRBN
        network = DRBN()
        
    elif network_type == 'CIDNet':
        from models.CIDNet.CIDNet import CIDNet
        network = CIDNet()

    network.load_state_dict(torch.load(weight_path))

    return network
