import argparse
import os
import re

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torchvision
# from torchvision.utils import save_image
import yaml
# from ignite.contrib import metrics
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, eval_seg_pro, compute_AUAnomaly_Curve
# import constants as const
import dataset
from models import build_model
import utils
from visual import show_anomaly
from dataset import _CLASSNAMES, _SCENARIO
import random
from main import setup_seed, build_test_data_loader, parse_args, build_optimizer
import torch.nn.functional as F
# from torchvision.utils import save_image


def build_train_data_loader(args, config, broken):
    train_dataset = dataset.MVTecDataset_Scenario_GLASS(
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
    model.forward_modules.eval()
    if model.pre_proj > 0:
        model.pre_projection.train()
    model.discriminator.train()

    loss_meter = utils.AverageMeter()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    masked_img_prev = None

    for i_iter, data_item in enumerate(pbar, start=1):
        model.dsc_opt.zero_grad()
        if model.pre_proj > 0:
            model.proj_opt.zero_grad()

        aug = data_item["aug"]
        aug = aug.to(torch.float).to(device)
        
        img = data_item["image"]
        img = img.to(torch.float).to(device)

        if masked_img_prev is not None:
            beta = 0.5
            apply_mask = ((masked_img_prev != 0) & (torch.rand(B, 1, 1, 1, device=img.device) < 0.5))
            # img = torch.where(apply_mask, masked_img_prev, img)
            img = torch.where(apply_mask, (1-beta)*img + beta*masked_img_prev, img)
            masked_img_prev = None

        if model.pre_proj > 0:
            fake_feats = model.pre_projection(model._embed(aug, evaluation=False)[0])
            fake_feats = fake_feats[0] if len(fake_feats) == 2 else fake_feats
            true_feats = model.pre_projection(model._embed(img, evaluation=False)[0])
            true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
        else:
            fake_feats = model._embed(aug, evaluation=False)[0]
            fake_feats.requires_grad = True
            true_feats = model._embed(img, evaluation=False)[0]
            true_feats.requires_grad = True

        # (b, 32, 32)
        B, h_f, w_f = data_item["mask_s"].shape
        mask_s_gt = data_item["mask_s"].reshape(-1, 1).to(device)
        noise = torch.normal(0, model.noise, true_feats.shape).to(device)
        gaus_feats = true_feats + noise

        center = model.c.repeat(img.shape[0], 1, 1)
        center = center.reshape(-1, center.shape[-1])

        # augされた特徴（fake feats）の正常領域と、何もしなかった正常領域を集める
        true_points = torch.concat([fake_feats[mask_s_gt[:, 0] == 0], true_feats], dim=0)
        
        # 正常分布の中心ベクトル　（augされた特徴の正常部分の中心＋全体の特徴の中心）
        c_t_points = torch.concat([center[mask_s_gt[:, 0] == 0], center], dim=0)

        # 中心ベクトル（正常特徴）からの正常ベクトルの距離
        dist_t = torch.norm(true_points - c_t_points, dim=1)

        # 正常特徴が「中心からどのくらい離れてよいか」の閾値を決めるもの　（球の半径）
        r_t = torch.tensor([torch.quantile(dist_t, q=model.radius)]).to(device)

        for step in range(model.step + 1):
            # 識別器に正常特徴と、ガウシアンで作った異常を入力
            scores = model.discriminator(torch.cat([true_feats, gaus_feats]))
            true_scores = scores[:len(true_feats)]
            gaus_scores = scores[len(true_feats):]
            true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
            bce_loss = true_loss + gaus_loss

            if masked_img_prev is None:
                true_segmap = F.interpolate(
                    true_scores.view(B, h_f, w_f).unsqueeze(1).detach().clone(),
                    size=(img.shape[-2], img.shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                )
                threshold = 0.8
                true_segmap = (true_segmap - true_segmap.min()) / (true_segmap.max() - true_segmap.min())
                hard_mask = (true_segmap > threshold).float()  # (B,1,H,W)
                masked_img = img * hard_mask  # (B,C,H,W)

                masked_img_prev = masked_img


            if step == model.step:
                break
            elif model.mining == 0:
                dist_g = torch.norm(gaus_feats - center, dim=1)
                r_g = torch.tensor([torch.quantile(dist_g, q=model.radius)]).to(device)
                break
            
            grad = torch.autograd.grad(gaus_loss, [gaus_feats])[0]
            grad_norm = torch.norm(grad, dim=1)
            grad_norm = grad_norm.view(-1, 1)
            # 正規化された勾配を取る
            grad_normalized = grad / (grad_norm + 1e-10)

            # 勾配ベクトルを少し大きくする（勾配上昇）
            with torch.no_grad():
                gaus_feats.add_(0.001 * grad_normalized)

            if (step + 1) % 5 == 0:
                # 正常特徴からの距離を計測する（ガウス異常特徴）
                dist_g = torch.norm(gaus_feats - center, dim=1)
                r_g = torch.tensor([torch.quantile(dist_g, q=model.radius)]).to(device)
                proj_feats = center if model.svd == 1 else true_feats
                r = r_t if model.svd == 1 else 0.5

                # 正常特徴からどのくらい離れているか
                h = gaus_feats - proj_feats

                h_norm = dist_g if model.svd == 1 else torch.norm(h, dim=1)
                # 最低r、最大2rとなるように異常特徴を調整
                alpha = torch.clamp(h_norm, r, 2 * r)
                # スケーリング
                proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
                # 元のhの範囲をスケーリングさせる
                h = proj * h
                # 半径rから2rの球に特徴が射影される（境界に異常が行って、それが異常として捉えられる）
                gaus_feats = proj_feats + h

        fake_points = fake_feats[mask_s_gt[:, 0] == 1]
        true_points = true_feats[mask_s_gt[:, 0] == 1]
        c_f_points = center[mask_s_gt[:, 0] == 1]
        dist_f = torch.norm(fake_points - c_f_points, dim=1)
        r_f = torch.tensor([torch.quantile(dist_f, q=model.radius)]).to(device)
        proj_feats = c_f_points if model.svd == 1 else true_points
        r = r_t if model.svd == 1 else 1

        if model.svd == 1:
            h = fake_points - proj_feats
            h_norm = dist_f if model.svd == 1 else torch.norm(h, dim=1)
            alpha = torch.clamp(h_norm, 2 * r, 4 * r)
            proj = (alpha / (h_norm + 1e-10)).view(-1, 1)
            h = proj * h
            fake_points = proj_feats + h
            fake_feats[mask_s_gt[:, 0] == 1] = fake_points

        fake_scores = model.discriminator(fake_feats)
        if model.p > 0:
            fake_dist = (fake_scores - mask_s_gt) ** 2
            d_hard = torch.quantile(fake_dist, q=model.p)
            fake_scores_ = fake_scores[fake_dist >= d_hard].unsqueeze(1)
            mask_ = mask_s_gt[fake_dist >= d_hard].unsqueeze(1)
        else:
            fake_scores_ = fake_scores
            mask_ = mask_s_gt
        output = torch.cat([1 - fake_scores_, fake_scores_], dim=1)
        focal_loss = model.focal_loss(output, mask_)

        loss = bce_loss + focal_loss
        loss.backward()
        if model.pre_proj > 0:
            model.proj_opt.step()
        if model.train_backbone:
            model.backbone_opt.step()
        model.dsc_opt.step()

        # log
        loss_meter.update(loss.item())
        del loss
        if (step + 1) % config["data"]["LOG_INTERVAL"] == 0 or (step + 1) == len(dataloader):
            pbar.set_description(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )
    

def eval_once(test_dataloader, model, args, broken, device):

    model.eval()

    anomaly_types = []
    labels = []
    image_scores_ = []
    pixel_scores_ = []

    idx = 0
    anoma_max = 0.
    anoma_min = 1.

    for anomaly, data, targets in tqdm(test_dataloader, total=len(test_dataloader)):

        data, targets = data.cuda(device), targets.cuda(device)

        patch_features, patch_shapes = model._embed(data, provide_patch_shapes=True, evaluation=True)
        if model.pre_proj > 0:
            patch_features = model.pre_projection(patch_features)
            patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

        patch_scores = image_scores = model.discriminator(patch_features)
        patch_scores = model.patch_maker.unpatch_scores(patch_scores, batchsize=data.shape[0])
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(data.shape[0], scales[0], scales[1])
        masks = model.anomaly_segmentor.convert_to_segmentation(patch_scores)

        image_scores = model.patch_maker.unpatch_scores(image_scores, batchsize=data.shape[0])
        image_scores = model.patch_maker.score(image_scores)
        if isinstance(image_scores, torch.Tensor):
            image_scores = image_scores.detach().cpu().numpy()
        masks = [torch.from_numpy(mask).to(device) for mask in masks]
        masks = torch.stack(masks, 0)

        if masks.max() > anoma_max : anoma_max = masks.max().item()
        if masks.min() < anoma_min : anoma_min = masks.min().item()
        
        # show_anomaly(data, targets, masks.unsqueeze(1), f"{args.result}/_{args.scenario}-exp_{args.category}-{broken}/{idx}.png", limit=[anoma_min, anoma_max])
        idx += 1
        anomaly_types.extend(anomaly)
        image_scores_.append(image_scores)
        pixel_scores_.append(masks.cpu().detach().numpy())
        targets = (targets != 0) * 1.0
        labels.append(targets.cpu().detach().numpy())
    
    # 画像ごとの pixel-スコアをまとめる
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
            model.forward_modules.eval()
            with torch.no_grad():  # compute center
                for i, data in enumerate(train_dataloader):
                    img = data["image"]
                    img = img.to(torch.float).to(device)
                    if model.pre_proj > 0:
                        outputs = model.pre_projection(model._embed(img, evaluation=False)[0])
                        outputs = outputs[0] if len(outputs) == 2 else outputs
                    else:
                        outputs = model._embed(img, evaluation=False)[0]
                    outputs = outputs[0] if len(outputs) == 2 else outputs
                    outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])

                    batch_mean = torch.mean(outputs, dim=0)
                    if i == 0:
                        model.c = batch_mean
                    else:
                        model.c += batch_mean
                model.c /= len(train_dataloader)
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
        config = yaml.safe_load(open(args.config, "r"))
        method, _ = os.path.splitext(os.path.basename(args.config))
        model = build_model(config, method, args.category, device)
        checkpoint = torch.load(f"{args.result}/_{args.scenario}-exp_{args.category}-{broken}/best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        test_dataloader = build_test_data_loader(args, config, broken)
        model.cuda(device)
        result_dict = eval_once(test_dataloader, model, args, broken, device)
        
        result_path = os.path.join(args.result, f"_{args.scenario}-exp_{args.category}-{broken}", "_result.txt")
        # with open(result_path, "w") as f:
        #     for k, v in result_dict.items():
        #         f.write(f"{k}\t {v:.2%}\n")

if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    print(f"category:{args.category}")
    # vis_std(args)
    if args.eval:
        evaluate(args)
    else:
        train(args)
