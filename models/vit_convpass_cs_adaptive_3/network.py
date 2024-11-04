from .timm_vit import _create_vit_adapter, _create_vit
import logging
import timm
from timm.models.vision_transformer import vit_base_patch16_224
from .convpass import set_Convpass




def build_net(arch_name, pretrained, dim, adapter_num, **kwargs):
    if arch_name == 'vit_base_patch16_224':
        # model = vit_base_patch16_224(pretrained, num_classes=kwargs['num_classes'])
        model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,)
        model = _create_vit('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)

    if kwargs['conv_type'] == 'cdc':
        set_Convpass(model, 'convpass', dim=dim, s=1, xavier_init=False, conv_type=kwargs['conv_type'])
    else:
        set_Convpass(model, 'convpass', dim=dim, adapter_num=adapter_num, s=1, xavier_init=True, conv_type=kwargs['conv_type'])
    #import pdb; pdb.set_trace()

    return model

if __name__ == '__main__':
    build_net('vit_base_patch16_224')

