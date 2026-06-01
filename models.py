import torch

def build_model(config, method, category, device):

    if method == "fastflow":
        import model.FastFlow.fastflow as fastflow
        model = fastflow.FastFlow(
            backbone_name=config["model"]["backbone_name"],
            flow_steps=config["model"]["flow_step"],
            input_size=config["data"]["input_size"],
            conv3x3_only=config["model"]["conv3x3_only"],
            hidden_ratio=config["model"]["hidden_ratio"],
        )
    elif method == "mambaAD":
        from model.MambaAD.model.mambaad import MAMBAAD 
        model = MAMBAAD(config)
    elif method == "Dinomaly" or method == "Dinomaly_pro":
        from model.Dinomaly.dinomaly_model import DinomalyModel
        model = DinomalyModel(config, device)
    elif method == "INP_Former":
        import torch.nn as nn
        from functools import partial
        from model.INP_Former.models import vit_encoder
        from model.INP_Former.models.uad import INP_Former
        from model.INP_Former.models.vision_transformer import Mlp, Aggregation_Block, Prototype_Block

        target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
        fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
        encoder = vit_encoder.load(config["model"]["encoder"])
        if 'small' in config["model"]["encoder"]:
            embed_dim, num_heads = 384, 6
        elif 'base' in config["model"]["encoder"]:
            embed_dim, num_heads = 768, 12
        elif 'large' in config["model"]["encoder"]:
            embed_dim, num_heads = 1024, 16
            target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
        else:
            raise "Architecture not in small, base, large."
        
        # Model Preparation
        Bottleneck = []
        INP_Guided_Decoder = []
        INP_Extractor = []

        # bottleneck
        Bottleneck.append(Mlp(embed_dim, embed_dim * 4, embed_dim, drop=0.))
        Bottleneck = nn.ModuleList(Bottleneck)

        # INP
        INP = nn.ParameterList(
                        [nn.Parameter(torch.randn(config["model"]["INP_num"], embed_dim))
                        for _ in range(1)])
        # INP Extractor
        for i in range(1):
            blk = Aggregation_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                    qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            INP_Extractor.append(blk)
        INP_Extractor = nn.ModuleList(INP_Extractor)

        # INP_Guided_Decoder
        for i in range(8):
            blk = Prototype_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.,
                                qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-8))
            INP_Guided_Decoder.append(blk)
        INP_Guided_Decoder = nn.ModuleList(INP_Guided_Decoder)

        model = INP_Former(encoder=encoder, bottleneck=Bottleneck, aggregation=INP_Extractor, decoder=INP_Guided_Decoder,
                             target_layers=target_layers,  remove_class_token=True, fuse_layer_encoder=fuse_layer_encoder,
                             fuse_layer_decoder=fuse_layer_decoder, prototype_token=INP)
    elif method == "UniNet":

        from model.UniNet.UniNet_lib.resnet import wide_resnet50_2
        from model.UniNet.UniNet_lib.de_resnet import de_wide_resnet50_2
        from model.UniNet.UniNet_lib.model import UniNet
        from model.UniNet.UniNet_lib.DFS import DomainRelated_Feature_Selection
        from model.UniNet.utils import to_device
        import copy

        Source_teacher, bn = wide_resnet50_2(c=config, pretrained=True)
        Source_teacher.layer4 = None
        Source_teacher.fc = None
        student = de_wide_resnet50_2(pretrained=False)
        DFS = DomainRelated_Feature_Selection()
        [Source_teacher, bnu, stdent, DFS] = to_device([Source_teacher, bn, student, DFS], device)
        Target_teacher = copy.deepcopy(Source_teacher)
        
        # params = list(student.parameters()) + list(bn.parameters()) + list(DFS.parameters())
        model = UniNet(config, Source_teacher, Target_teacher, bn, student, category, DFS=DFS)
    elif method == "RDplus2":
        from model.RDplus2.RDplus2 import RDPLUS2
        model = RDPLUS2(config)
    elif method == "RD4AD":
        from model.RD4AD.RD4AD_model import RD4AD
        model = RD4AD(config)
    elif method == "DiAD":
        from model.DiAD.DiAD_model import DiAD_Model
        model = DiAD_Model(config, device)
    elif method == "GLASS" or method == "RePaste":
        from model.GLASS import glass, backbones
        model = glass.GLASS(device)
        layers_str = config["model"]["layers_to_extract_from"]  # "layer2 layer3"
        layers_list = layers_str.split()  # ['layer2', 'layer3']
        model.load(
            backbone=backbones.load(config["model"]["backbone_name"]),
            layers_to_extract_from=layers_list,
            device=device,
            input_shape=config["data"]["input_size_train"],
            pretrain_embed_dimension=config["model"].get("pretrain_embed_dimension", 1536),
            target_embed_dimension=config["model"].get("target_embed_dimension", 1536),
            patchsize=config["model"].get("patchsize", 3),
            meta_epochs=config["data"].get("NUM_EPOCHS", 640),
            eval_epochs=config["data"].get("EVAL_INTERVAL", 1),
            dsc_layers=config["model"].get("dsc_layers", 2),
            dsc_hidden=config["model"].get("dsc_hidden", 1024),
            dsc_margin=config["model"].get("dsc_margin", 0.5),
            train_backbone=config["model"].get("train_backbone", False),
            pre_proj=config["model"].get("pre_proj", 1),
            mining=config["model"].get("mining", 1),
            noise=config["model"].get("noise", 0.015),
            radius=config["model"].get("radius", 0.75),
            p=config["model"].get("p", 0.5),
            lr=config["model"].get("lr", 1e-4),
            svd=config["model"].get("svd", 0),
            step=config["model"].get("step", 20),
            limit=config["model"].get("limit", 392),
        )
        model = model.to(device)
        print(f'svd: {config["model"].get("svd", 0)}')
    elif method == "GLASS_pro_normal_plus":
        from model.GLASS import glass_noise, backbones
        model = glass_noise.GLASS(device)
        layers_str = config["model"]["layers_to_extract_from"]  # "layer2 layer3"
        layers_list = layers_str.split()  # ['layer2', 'layer3']
        model.load(
            backbone=backbones.load(config["model"]["backbone_name"]),
            layers_to_extract_from=layers_list,
            device=device,
            input_shape=config["data"]["input_size_train"],
            pretrain_embed_dimension=config["model"].get("pretrain_embed_dimension", 1536),
            target_embed_dimension=config["model"].get("target_embed_dimension", 1536),
            patchsize=config["model"].get("patchsize", 3),
            meta_epochs=config["data"].get("NUM_EPOCHS", 640),
            eval_epochs=config["data"].get("EVAL_INTERVAL", 1),
            dsc_layers=config["model"].get("dsc_layers", 2),
            dsc_hidden=config["model"].get("dsc_hidden", 1024),
            dsc_margin=config["model"].get("dsc_margin", 0.5),
            train_backbone=config["model"].get("train_backbone", False),
            pre_proj=config["model"].get("pre_proj", 1),
            mining=config["model"].get("mining", 1),
            noise=config["model"].get("noise", 0.015),
            radius=config["model"].get("radius", 0.75),
            p=config["model"].get("p", 0.5),
            lr=config["model"].get("lr", 1e-4),
            svd=config["model"].get("svd", 0),
            step=config["model"].get("step", 20),
            limit=config["model"].get("limit", 392),
        )
        model = model.to(device)
        print(f'svd: {config["model"].get("svd", 0)}')
    
    elif method == "SimpleNet" or method == "SimpleNet_pro":
        from model.SimpleNet import simplenet, backbones
        model = simplenet.SimpleNet(device)
        layers_str = config["model"]["layers_to_extract_from"]  # "layer2 layer3"
        layers_list = layers_str.split()  # ['layer2', 'layer3']
        model.load(
            backbone=backbones.load(config["model"]["backbone_name"]),
            layers_to_extract_from=layers_list,
            device=device,
            input_shape=config["data"]["input_size_train"],
            pretrain_embed_dimension=config["model"].get("pretrain_embed_dimension", 1536),
            target_embed_dimension=config["model"].get("target_embed_dimension", 1536),
            patchsize=config["model"].get("patchsize", 3),
            embedding_size=config["model"].get("embedding_size", 256), #
            meta_epochs=config["data"].get("NUM_EPOCHS", 40),
            gan_epochs=config["data"].get("gan_epochs", 4),
            noise_std=config["data"].get("noise_std", 0.015),
            dsc_layers=config["model"].get("dsc_layers", 2),
            dsc_hidden=config["model"].get("dsc_hidden", 1024),
            dsc_margin=config["model"].get("dsc_margin", 0.5),
            dsc_lr=config["model"].get("dsc_lr", 0.0002),
            train_backbone=config["model"].get("train_backbone", False),
            cos_lr=config["model"].get("cos_lr", True),
            pre_proj=config["model"].get("pre_proj", 1),
            lr=config["model"].get("lr", 1e-3),
        )
        model = model.to(device)
    
    elif method == "patchcore":
        import model.PatchCore.src.patchcore.backbones as backbones
        import model.PatchCore.src.patchcore.common as common
        import model.PatchCore.src.patchcore.patchcore as patchcore
        import model.PatchCore.src.patchcore.sampler as sampler

        backbone_name = config["model"]["backbone_name"]
        backbone_seed = None
        if ".seed-" in backbone_name:
            backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
        
        backbone = backbones.load(backbone_name)
        backbone.name, backbone.seed = backbone_name, backbone_seed
        
        percentage = config["model"].get("coreset_sampling_ratio", 0.1)
        sampler = sampler.ApproximateGreedyCoresetSampler(percentage, device)
        
        faiss_on_gpu = config["model"].get("faiss_on_gpu", False)
        faiss_num_workers = config["model"].get("faiss_num_workers", 8)
        nn_method = common.FaissNN(faiss_on_gpu, faiss_num_workers)
        
        model = patchcore.PatchCore(device)
        
        layers_str = config["model"]["layers_to_extract_from"]
        layers_list = layers_str.split() if isinstance(layers_str, str) else layers_str

        model.load(
            backbone=backbone,
            layers_to_extract_from=layers_list,
            device=device,
            input_shape=config["data"]["input_size"],
            pretrain_embed_dimension=config["model"].get("pretrain_embed_dimension", 1024),
            target_embed_dimension=config["model"].get("target_embed_dimension", 1024),
            patchsize=config["model"].get("patchsize", 3),
            featuresampler=sampler,
            anomaly_scorer_num_nn=config["model"].get("anomaly_scorer_num_nn", 5),
            nn_method=nn_method,
        )
    else:
        raise NameError(f"Unknown method: {method}")

    print(
        "Model A.D. Param#: {}".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    return model