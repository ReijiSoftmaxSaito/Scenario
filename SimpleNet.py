import argparse
import os
import re

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torchvision
from torchvision.utils import save_image
import yaml
# from ignite.contrib import metrics
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, eval_seg_pro, compute_AUAnomaly_Curve
import constants as const
import dataset
from models import build_model
import utils
from visual import show_anomaly
from dataset import _CLASSNAMES, _SCENARIO
from main import setup_seed, build_test_data_loader, parse_args, build_optimizer


def build_train_data_loader(args, config, broken):
    train_dataset = dataset.MVTecDataset_Scenario_SimpleNet(
        root=args.data,
        category=args.category,
        scenario=args.scenario,
        broken_type=broken,
        config=config,
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["data"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

def train_one_epoch(dataloader, model, epoch, config, device, **kwargs):
    """Computes and sets the support features for SPADE."""
    _ = model.forward_modules.eval()
    
    if model.pre_proj > 0:
        model.pre_projection.train()
    model.discriminator.train()
    # self.feature_enc.eval()
    # self.feature_dec.eval()
    i_iter = 0
    # LOGGER.info(f"Training discriminator...")
    loss_meter = utils.AverageMeter()
    with tqdm(total=model.gan_epochs) as pbar:
        for i_epoch in range(model.gan_epochs):
            embeddings_list = []
            for data_item in dataloader:
                model.dsc_opt.zero_grad()
                if model.pre_proj > 0:
                    model.proj_opt.zero_grad()
                # self.dec_opt.zero_grad()

                i_iter += 1
                img = data_item["image"]
                img = img.to(torch.float).to(device)
                if model.pre_proj > 0:
                    true_feats = model.pre_projection(model._embed(img, evaluation=False)[0])
                else:
                    true_feats = model._embed(img, evaluation=False)[0]
                
                noise_idxs = torch.randint(0, model.mix_noise, torch.Size([true_feats.shape[0]]))
                noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=model.mix_noise).to(device) # (N, K)
                noise = torch.stack([
                    torch.normal(0, model.noise_std * 1.1**(k), true_feats.shape)
                    for k in range(model.mix_noise)], dim=1).to(device) # (N, K, C)
                noise = (noise * noise_one_hot.unsqueeze(-1)).sum(1)
                fake_feats = true_feats + noise

                scores = model.discriminator(torch.cat([true_feats, fake_feats]))
                true_scores = scores[:len(true_feats)]
                fake_scores = scores[len(fake_feats):]
                
                th = model.dsc_margin
                # p_true = (true_scores.detach() >= th).sum() / len(true_scores)
                # p_fake = (fake_scores.detach() < -th).sum() / len(fake_scores)
                true_loss = torch.clip(-true_scores + th, min=0)
                fake_loss = torch.clip(fake_scores + th, min=0)

             
                loss = true_loss.mean() + fake_loss.mean()
            
                loss.backward()
                if model.pre_proj > 0:
                    model.proj_opt.step()
                if model.train_backbone:
                    model.backbone_opt.step()
                model.dsc_opt.step()

                loss = loss.detach().cpu() 
                # all_loss.append(loss.item())
                # all_p_true.append(p_true.cpu().item())
                # all_p_fake.append(p_fake.cpu().item())
                loss_meter.update(loss.item())
            
            if len(embeddings_list) > 0:
                model.auto_noise[1] = torch.cat(embeddings_list).std(0).mean(-1)
            
            if model.cos_lr:
                model.dsc_schl.step()

            # log
            
            pbar.update(1)
            del loss
            if (model.gan_epochs + 1) % config["data"]["LOG_INTERVAL"] == 0 or (model.gan_epochs + 1) == len(dataloader):
                pbar.set_description(
                    "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                        epoch + 1, model.gan_epochs + 1, loss_meter.val, loss_meter.avg
                    )
                )
        

def eval_once(test_dataloader, model, args, broken, device):

    _ = model.forward_modules.eval()

    anomaly_types = []
    labels = []
    image_scores_ = []
    pixel_scores_ = []

    idx = 0
    anoma_max = 0.
    anoma_min = 1.

    if model.pre_proj > 0:
        model.pre_projection.eval()
    model.discriminator.eval()

    for anomaly, data, targets in tqdm(test_dataloader, total=len(test_dataloader)):
        with torch.no_grad():
            data, targets = data.cuda(device), targets.cuda(device)
            batchsize = data.shape[0]
            features, patch_shapes = model._embed(data,
                                                    provide_patch_shapes=True, 
                                                    evaluation=True)
            if model.pre_proj > 0:
                features = model.pre_projection(features)

            # features = features.cpu().numpy()
            # features = np.ascontiguousarray(features.cpu().numpy())
            patch_scores = image_scores = -model.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = model.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            image_scores = model.patch_maker.score(image_scores)

            patch_scores = model.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
            features = features.reshape(batchsize, scales[0], scales[1], -1)
            masks, features = model.anomaly_segmentor.convert_to_segmentation(patch_scores, features)
        
        masks = [torch.from_numpy(mask).to(device) for mask in masks]
        masks = torch.stack(masks, 0)
        if masks.max() > anoma_max : anoma_max = masks.max().item()
        if masks.min() < anoma_min : anoma_min = masks.min().item()
        
        show_anomaly(data, targets, masks.unsqueeze(1), f"{args.result}/_{args.scenario}-exp_{args.category}-{broken}/{idx}.png", limit=[anoma_min, anoma_max])
        idx += 1
        anomaly_types.extend(anomaly)
        image_scores_.append(image_scores)
        pixel_scores_.append(masks.cpu().detach().numpy())
        targets = (targets != 0) * 1.0
        labels.append(targets.cpu().detach().numpy())
    
    image_scores = np.concatenate(image_scores_, axis=0)
    pixel_scores = np.concatenate(pixel_scores_, axis=0) 
    
    labels = np.concatenate(labels, axis=0)
    image_labels = labels.reshape(labels.shape[0], -1).max(axis=-1)

    pixel_labels = labels[:,0]

    if test_dataloader.dataset.scenario == "A2N":
        anomaly_flag = {test_dataloader.dataset.broken_type:[test_dataloader.dataset.broken_type in ano for ano in anomaly_types]}
    elif test_dataloader.dataset.scenario == "N2A":
        anomaly_flag = {"pseudo_anomaly":["pseudo_anomaly" in ano for ano in anomaly_types]}
    elif test_dataloader.dataset.scenario == "Normal" and "good-pseudo_anomaly" in anomaly_types: # N2A Normal
        anomaly_flag = {"pseudo_anomaly":["pseudo_anomaly" in ano for ano in anomaly_types]}
    else: #A2N Normal
        anomaly_flag = {broken_type:[broken_type in ano for ano in anomaly_types] for broken_type in _SCENARIO[test_dataloader.dataset.category]}

    imagewize_AUROC = compute_imagewise_retrieval_metrics(image_scores, image_labels)
    pixelwize_AUROC = compute_pixelwise_retrieval_metrics(pixel_scores, pixel_labels)
    PRO = eval_seg_pro(pixel_labels, pixel_scores)
    OtherL1Score = {}
    for broken_type, flag in anomaly_flag.items():
        flag = np.array(flag)
        image_labels = np.array(image_labels)

        min_scores = image_scores.min(axis=-1).reshape(-1, 1)
        max_scores = image_scores.max(axis=-1).reshape(-1, 1)
        image_scores = (image_scores - min_scores) / (max_scores - min_scores)
        image_scores = np.mean(image_scores, axis=0)
        OtherL1Score[broken_type + "-OtherL1Score"] = compute_imagewise_retrieval_metrics(image_scores[~flag], image_labels[~flag])["auroc"]
    
    OnlyL1Score = {}
    for broken_type, flag in anomaly_flag.items():
        flag = np.array(flag)
        image_labels = np.array(image_labels)

        min_scores = image_scores.min(axis=-1).reshape(-1, 1)
        max_scores = image_scores.max(axis=-1).reshape(-1, 1)
        image_scores = (image_scores - min_scores) / (max_scores - min_scores)
        image_scores = np.mean(image_scores, axis=0)

        OnlyL1Score[broken_type + "-OnlyL1Score"] = compute_AUAnomaly_Curve(image_scores[flag])

    print("Image level AUROC: {:.2%}".format(imagewize_AUROC["auroc"]))
    print("Pixel level AUROC: {:.2%}".format(pixelwize_AUROC["auroc"]))
    print("PRO: {:.2%}".format(PRO))
    for key, value in OnlyL1Score.items():
        print(f"{key}: {value:.2%}")
    for key, value in OtherL1Score.items():
        print(f"{key}: {value:.2%}")

    return {"Image_level_AUROC":imagewize_AUROC["auroc"], "Pixel_level_AUROC":pixelwize_AUROC["auroc"], "PRO": PRO, **OnlyL1Score, **OtherL1Score}

def train(args):
    if "Normal" in args.scenario:
        A2N_N2A = args.scenario.split("_")[0]
        args.scenario = args.scenario.split("_")[1]
    else:
        A2N_N2A = args.scenario

    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    args.result = f"result/{args.result}/{A2N_N2A}"

    if args.scenario == "Normal":
        broken_type = ["None"]
    elif args.scenario == "A2N":
        broken_type = _SCENARIO[args.category]
    elif args.scenario == "N2A":
        broken_type = ["pseudo_anomaly"]
    
    for broken in broken_type:
        os.makedirs(args.result, exist_ok=True)
        checkpoint_dir = os.path.join(
            args.result, f"_{args.scenario}-exp_{args.category}-{broken}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        config = yaml.safe_load(open(args.config, "r"))
        method, _ = os.path.splitext(os.path.basename(args.config))
        model = build_model(config, method, args.category, device)
        # optimizer = build_optimizer(model, config)

        train_dataloader = build_train_data_loader(args, config, broken)
        test_dataloader = build_test_data_loader(args, config, broken)

            
        model.cuda(device)

        best_score = 0.0
        best_epoch = 0
        best_result = {"temp",0.0}
        for epoch in range(config["data"]["NUM_EPOCHS"]):
            train_one_epoch(train_dataloader, model, epoch, config, device)
            if (epoch + 1) % config["data"]["EVAL_INTERVAL"] == 0:
                eval_score = eval_once(test_dataloader, model, args, broken, device)
                
                if best_score <= np.mean(list(eval_score.values())[:3]):
                    best_epoch = epoch
                    best_score = np.mean(list(eval_score.values())[:3])
                    best_result = eval_score
                    torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                # "optimizer_state_dict": optimizer[0].state_dict(),
                            },
                            os.path.join(checkpoint_dir, "best.pt"),
                        )

        
        result_path = os.path.join(args.result, f"_{args.scenario}-exp_{args.category}-{broken}", "_best_epoch.txt")
        with open(result_path, "w") as f:
            f.write(f"best epoch: {best_epoch:d}")
        
        result_path = os.path.join(args.result, f"_{args.scenario}-exp_{args.category}-{broken}", "_result.txt")
        with open(result_path, "w") as f:
            for k, v in best_result.items():
                f.write(f"{k}\t {v:.2%}\n")

def evaluate(args):
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    args.result = f"result/{args.result}"
    if args.scenario == "Normal":
        broken_type = ["None"]
    elif args.scenario == "A2N":
        broken_type = _SCENARIO[args.category]
    elif args.scenario == "N2A":
        broken_type = ["pseudo_anomaly"]

    train_dataloader = None
    train_ = False
        
    for broken in broken_type:
        config = yaml.safe_load(open(args.config, "r"))
        method, _ = os.path.splitext(os.path.basename(args.config))
        model = build_model(config, method, device)
        checkpoint = torch.load(f"{args.result}/_{args.scenario}-exp_{args.category}-{broken}/best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        test_dataloader = build_test_data_loader(args, config, broken)
        model.cuda(device)
        result_dict = eval_once(test_dataloader, model, args, broken, device)
        
        result_path = os.path.join(args.result, f"_{args.scenario}-exp_{args.category}-{broken}", "_result.txt")
        with open(result_path, "w") as f:
            for k, v in result_dict.items():
                f.write(f"{k}\t {v:.2%}\n")


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    print(f"category:{args.category}")
    # vis_std(args)
    if args.eval:
        evaluate(args)
    else:
        train(args)
