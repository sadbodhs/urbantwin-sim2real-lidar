"""Add FSHNet (utl/ext3p/fshnet, github.com/Say2L/FSHNet) and SAFDNet (utl/ext3p/safdnet, github.com/zhanggang001/HEDNet) to OpenPCDet's registries.
Only when torch_scatter is installed (image urbantwin-lidar:scatter); on urbantwin-lidar:latest nothing is imported or changed.
Voxel Mamba (utl/ext3p/voxel_mamba, github.com/gwenzhang/Voxel-Mamba) is added too when mamba_ssm is installed
(image urbantwin-lidar:mamba only); on urbantwin-lidar:scatter nothing about it is imported.
CenterPointAFD is the CenterPoint of both repos: the backbone also gets class_names, and the training loss adds the backbone's
adaptive-feature-diffusion loss (their pcdet/models/detectors/centerpoint.py and detector3d_template.py). Their other template changes only
remap pretrained weights, which are not used."""
import importlib.util


def register_ext():
    if importlib.util.find_spec("torch_scatter") is None:
        return False
    import pcdet.models.backbones_3d as B
    import pcdet.models.dense_heads as D
    import pcdet.models.detectors as M
    from pcdet.models.detectors.centerpoint import CenterPoint

    from .fshnet.fshnet_argo2 import FSHNet_argo2
    from .fshnet.sparse_dynamic_head import SparseDynamicHead
    from .fshnet.sparse_dynamic_head_multi import SparseDynamicHead_multi
    from .safdnet.hednet import SparseHEDNet
    from .safdnet.sparse_center_head import SparseCenterHead

    class CenterPointAFD(CenterPoint):
        def build_backbone_3d(self, model_info_dict):
            if self.model_cfg.get("BACKBONE_3D", None) is None:
                return None, model_info_dict
            m = B.__all__[self.model_cfg.BACKBONE_3D.NAME](
                model_cfg=self.model_cfg.BACKBONE_3D, input_channels=model_info_dict["num_point_features"],
                grid_size=model_info_dict["grid_size"], voxel_size=model_info_dict["voxel_size"],
                point_cloud_range=model_info_dict["point_cloud_range"], class_names=self.class_names)
            model_info_dict["module_list"].append(m)
            model_info_dict["num_point_features"] = m.num_point_features
            model_info_dict["backbone_channels"] = getattr(m, "backbone_channels", None)
            return m, model_info_dict

        def get_training_loss(self):
            loss, tb_dict, disp_dict = super().get_training_loss()
            if getattr(self.backbone_3d, "adaptive_feature_diffusion", False):
                loss_afd, tb_afd = self.backbone_3d.get_loss()
                loss = loss + loss_afd
                tb_dict.update(tb_afd)
            return loss, tb_dict, disp_dict

    for reg, name, obj in ((B, "FSHNet_argo2", FSHNet_argo2), (B, "SparseHEDNet", SparseHEDNet), (D, "SparseDynamicHead", SparseDynamicHead), (D, "SparseDynamicHead_multi", SparseDynamicHead_multi),
                           (D, "SparseCenterHead", SparseCenterHead), (M, "CenterPointAFD", CenterPointAFD)):
        assert reg.__all__.get(name, obj) is obj, f"{name} already registered by something else"
        reg.__all__[name] = obj
    if importlib.util.find_spec("mamba_ssm") is not None:
        from .voxel_mamba.voxel_mamba_waymo import Voxel_Mamba_Waymo
        assert B.__all__.get("Voxel_Mamba_Waymo", Voxel_Mamba_Waymo) is Voxel_Mamba_Waymo, "Voxel_Mamba_Waymo already registered by something else"
        B.__all__["Voxel_Mamba_Waymo"] = Voxel_Mamba_Waymo
    return True
