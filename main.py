import argparse
import os
import re

from tqdm import tqdm
import numpy as np
import cv2
import torch
import torchvision
import yaml
# from ignite.contrib import metrics
from metrics import compute_imagewise_retrieval_metrics, compute_pixelwise_retrieval_metrics, eval_seg_pro, compute_AUAnomaly_Curve
import constants as const
import dataset
from models import build_model
import utils
from visual import show_anomaly
from dataset import _CLASSNAMES, _SCENARIO
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_train_data_loader(args, config, broken, use_noise=False):
    train_dataset = dataset.MVTecDataset_Scenario(
        root=args.data,
        category=args.category,
        scenario=args.scenario,
        broken_type=broken,
        input_size=config["data"]["input_size"],
        use_noise=use_noise,
        is_train=True,
    )
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["data"]["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )


def build_test_data_loader(args, config, broken):

    test_dataset = dataset.MVTecDataset_Scenario(
        root=args.data,
        category=args.category,
        scenario=args.scenario,
        broken_type=broken,
        input_size=config["data"]["input_size"],
        is_train=False,
    )
    batch_size = config["data"].get("TEST_BATCH_SIZE", 8) 
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

def build_optimizer(model, method, category, config):
    if method == "UniNet":
        params = list(model.s.parameters()) \
               + list(model.bn.parameters()) \
               + list(model.dfs.parameters())
        optimizer_s = torch.optim.AdamW(
            params,
            lr=float(config["model"]["LR_S"]),
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        optimizer_t = torch.optim.AdamW(
            list(model.t.parameters()),
            lr=1e-4 if category == 'transistor' else float(config["model"]["LR_T"]),
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        optimizers = [optimizer_s]
        if optimizer_t is not None:
            optimizers.append(optimizer_t)

        return optimizers
    elif method == "RDplus2":
        
        optimizer_proj = torch.optim.Adam(list(model.proj_layer.parameters()), lr=config["model"]["LR_PROJ"], betas=(0.5,0.999))
        optimizer_distill = torch.optim.Adam(list(model.decoder.parameters())+list(model.bn.parameters()), lr=config["model"]["LR_DIST"], betas=(0.5,0.999))

        optimizers = [optimizer_proj]
        optimizers.append(optimizer_distill)
        
        return optimizers
    elif method == "Dinomaly":
        from model.Dinomaly.optimizers import StableAdamW
        import torch.nn as nn
        model.trainable_modules = nn.ModuleList([model.trainable])
        optimizer = StableAdamW([{'params': model.trainable_modules.parameters()}],
                            lr=float(config["model"]["LR"]), betas=(0.9, 0.999), weight_decay=float(config["model"]["WEIGHT_DECAY"]), amsgrad=True, eps=1e-8)
        return [optimizer]
    
    optimizer_name = config["model"].get("OPTIMIZER", "Adam")
    optimizer_class = torch.optim.AdamW if optimizer_name == "AdamW" else torch.optim.Adam

    optimizer = optimizer_class(
        model.parameters(),
        lr=float(config["model"]["LR"]),
        weight_decay=float(config["model"]["WEIGHT_DECAY"]),
        betas=(
            float(config["model"].get("BETA1", 0.9)),
            float(config["model"].get("BETA2", 0.999))
        )
    )

    return [optimizer]

def forward_model(model, data, method, it=None):

    if method == "patchcore":
        with torch.no_grad():
            return model.predict(data)

    if method == "Dinomaly":
        return model(data, it=it) 
    return model(data)

def model_step(model, data, config, method, train, device, it=None):
    ret = forward_model(model, data, method, it)
    if train :
        return ret["loss"]
    return ret["anomaly_map"].cpu().detach()

def train_one_epoch(dataloader, model, optimizer, epoch, config, method, train, device, **kwargs):
    scheduler = kwargs.get("scheduler", None)  # scheduler があれば取得、なければ None
    if method == "patchcore": 
        model.fit(dataloader) 
        return
    
    model.train()
    it = 0
    loss_meter = utils.AverageMeter()
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, data in enumerate(pbar, start=1):
        # forward
        data1 = data[1].cuda(device)
        if len(data) == 3:
            data_noise = data[2].cuda(device)
            loss = model(data1, x_noise=data_noise)
        else:
            loss = model_step(model, data1, config, method, train, device, it)

        # backward
        for opt in optimizer:
            opt.zero_grad()

        loss.backward()

        for opt in optimizer:
            opt.step()

        if scheduler is not None:
            scheduler.step()

        # log
        loss_meter.update(loss.item())
        del loss
        if (step + 1) % config["data"]["LOG_INTERVAL"] == 0 or (step + 1) == len(dataloader):
            pbar.set_description(
                "Epoch {} - Step {}: loss = {:.3f}({:.3f})".format(
                    epoch + 1, step + 1, loss_meter.val, loss_meter.avg
                )
            )

def eval_once(config, train_dataloader, test_dataloader, model, args, broken, method, train, device):

    model.eval()

    anomaly_types = []
    predicts = []
    labels = []

    idx = 0
    anoma_max = 0.
    anoma_min = 1.
    image_scores = []

    for anomaly, data, targets in tqdm(test_dataloader, total=len(test_dataloader)):

        data, targets = data.cuda(device), targets.cuda(device)

        with torch.no_grad():
            outputs = model_step(model, data, config, method, train, device)

        if method == "patchcore":
            pixel_map = outputs["anomaly_map"]
            image_score = outputs["image_score"]

            # show_anomaly(
            #     data,
            #     targets,
            #     pixel_map,
            #     f"{args.result}/_{args.scenario}-exp_{args.category}-{broken}/{idx}.png",
            #     limit=[anoma_min, anoma_max]
            # )

            predicts.append(pixel_map.numpy())
            image_scores.append(image_score.numpy())
        else:
            if outputs.max() > anoma_max : anoma_max = outputs.max().item()
            if outputs.min() < anoma_min : anoma_min = outputs.min().item()
            # show_anomaly(data, targets, outputs, f"{args.result}/_{args.scenario}-exp_{args.category}-{broken}/{idx}.png", limit=[anoma_min, anoma_max])
            predicts.append(outputs.numpy())

        idx += 1
        anomaly_types.extend(anomaly)
        targets = (targets != 0) * 1.0
        labels.append(targets.cpu().detach().numpy())
    
    predicts = np.concatenate(predicts, axis=0)
    labels = np.concatenate(labels, axis=0)

    if method == "patchcore":
        image_scores = np.concatenate(image_scores, axis=0)
    else:
        image_scores = predicts.reshape(predicts.shape[0], -1).max(axis=-1)
    image_labels = labels.reshape(labels.shape[0], -1).max(axis=-1)
    pixel_scores = predicts[:,0]
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
        optimizer = build_optimizer(model, method, args.category, config)

        use_noise = True if method == "RDplus2" else False
        train_dataloader = build_train_data_loader(args, config, broken, use_noise=use_noise)
        test_dataloader = build_test_data_loader(args, config, broken)
        
        scheduler = None
        if "scheduler" in config["model"] and config["model"]["scheduler"] is not None:
            from utils import WarmCosineScheduler
            scheduler = WarmCosineScheduler(optimizer[0], base_value=float(config["model"]["scheduler"]["base_value"]), final_value=float(config["model"]["scheduler"]["final_value"]), total_iters=int(config["data"]["NUM_EPOCHS"])*len(train_dataloader),
                                            warmup_iters=int(config["model"]["scheduler"]["warmup_iters"]))
            
        model.cuda(device)

        best_score = 0.0
        best_epoch = 0
        best_result = {"temp",0.0}
        for epoch in range(config["data"]["NUM_EPOCHS"]):
            train = True
            train_one_epoch(train_dataloader, model, optimizer, epoch, config, method, train, device, scheduler=scheduler)
            if (epoch + 1) % config["data"]["EVAL_INTERVAL"] == 0:
                train = False
                eval_score = eval_once(config, train_dataloader, test_dataloader, model, args, broken, method, train, device)
                
                if best_score <= np.mean(list(eval_score.values())[:3]):
                    best_epoch = epoch
                    best_score = np.mean(list(eval_score.values())[:3])
                    best_result = eval_score
                    torch.save(
                            {
                                "epoch": epoch,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer[0].state_dict(),
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
    if "Normal" in args.scenario:
        A2N_N2A = args.scenario.split("_")[0]
        args.scenario = args.scenario.split("_")[1]
    else:
        A2N_N2A = args.scenario

    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    args.result = f"{args.result}/{A2N_N2A}"

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
        model = build_model(config, method, args.category, device)
        checkpoint = torch.load(f"result/{args.result}/_{args.scenario}-exp_{args.category}-{broken}/best.pt")
        model.load_state_dict(checkpoint["model_state_dict"])
        test_dataloader = build_test_data_loader(args, config, broken)
        model.cuda(device)
        result_dict = eval_once(config, train_dataloader, test_dataloader, model, args, broken, method, train, device)
        
        result_path = os.path.join(args.result, f"_{args.scenario}-exp_{args.category}-{broken}", "_result.txt")
        with open(result_path, "w") as f:
            for k, v in result_dict.items():
                f.write(f"{k}\t {v:.2%}\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument('--result', default = '')
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument("--scenario", type=str, required=True, help="path to mvtec folder")
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )

    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    print(f"category:{args.category}")
    # vis_std(args)
    if args.eval:
        evaluate(args)
    else:
        train(args)
