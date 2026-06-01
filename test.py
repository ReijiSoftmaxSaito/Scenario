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
import constants as const
import dataset
from models import build_model
import utils
from visual import show_anomaly
from dataset import _CLASSNAMES, _SCENARIO
import random
from sklearn.metrics import roc_auc_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

def forward_model(model, data, config, method, device, it=None):
    
    if method == "Dinomaly":
        ret = model(data, it=it) 
    else:
        ret = model(data)

    return ret["anomaly_map"].cpu().detach()

def eval_loop_simplenet(test_dataloader, model_1, model_2, device):
    _ = model_1.forward_modules.eval()
    _ = model_2.forward_modules.eval()

    if model_1.pre_proj > 0:
        model_1.pre_projection.eval()
    model_1.discriminator.eval()

    if model_2.pre_proj > 0:
        model_2.pre_projection.eval()
    model_2.discriminator.eval()

    idx = 0
    target_scores_m1 = [] 
    target_scores_m2 = []  

    for anomaly, data, targets in tqdm(test_dataloader, total=len(test_dataloader)):
        data, targets = data.cuda(device), targets.cuda(device)

        with torch.no_grad():
            batchsize = data.shape[0]

            features, patch_shapes = model_1._embed(data,
                                                    provide_patch_shapes=True, 
                                                    evaluation=True)
            if model_1.pre_proj > 0:
                features = model_1.pre_projection(features)

            patch_scores = image_scores = -model_1.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = model_1.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            s1 = model_1.patch_maker.score(image_scores)

            features, patch_shapes = model_2._embed(data,
                                                    provide_patch_shapes=True, 
                                                    evaluation=True)
            if model_2.pre_proj > 0:
                features = model_2.pre_projection(features)

            patch_scores = image_scores = -model_2.discriminator(features)
            patch_scores = patch_scores.cpu().numpy()
            image_scores = image_scores.cpu().numpy()

            image_scores = model_2.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            s2 = model_2.patch_maker.score(image_scores)

        for i, ano_type in enumerate(anomaly):
            if "good-" in ano_type or "pseudo_anomaly" in ano_type:
                target_scores_m1.append(s1[i])
                target_scores_m2.append(s2[i])
        idx += 1
    return target_scores_m1, target_scores_m2

def eval_loop_glass(test_dataloader, model_1, model_2, device):
    model_1.eval()
    model_2.eval()

    idx = 0
    target_scores_m1 = []  
    target_scores_m2 = []  

    for anomaly, data, targets in tqdm(test_dataloader, total=len(test_dataloader)):
        data, targets = data.cuda(device), targets.cuda(device)

        with torch.no_grad():
            patch_features, patch_shapes = model_1._embed(data, provide_patch_shapes=True, evaluation=True)
            if model_1.pre_proj > 0:
                patch_features = model_1.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = model_1.discriminator(patch_features)
            patch_scores = model_1.patch_maker.unpatch_scores(patch_scores, batchsize=data.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(data.shape[0], scales[0], scales[1])

            image_scores = model_1.patch_maker.unpatch_scores(image_scores, batchsize=data.shape[0])
            outputs_1 = model_1.patch_maker.score(image_scores)


            patch_features, patch_shapes = model_2._embed(data, provide_patch_shapes=True, evaluation=True)
            if model_2.pre_proj > 0:
                patch_features = model_2.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = model_2.discriminator(patch_features)
            patch_scores = model_2.patch_maker.unpatch_scores(patch_scores, batchsize=data.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(data.shape[0], scales[0], scales[1])

            image_scores = model_2.patch_maker.unpatch_scores(image_scores, batchsize=data.shape[0])
            outputs_2 = model_2.patch_maker.score(image_scores)

        s1 = outputs_1.reshape(data.shape[0], -1).max(axis=-1)[0].detach().cpu().numpy()
        s2 = outputs_2.reshape(data.shape[0], -1).max(axis=-1)[0].detach().cpu().numpy()

        for i, ano_type in enumerate(anomaly):
            if "good-" in ano_type or "pseudo_anomaly" in ano_type:
                target_scores_m1.append(s1[i])
                target_scores_m2.append(s2[i])

        idx += 1
    return target_scores_m1, target_scores_m2

def eval_loop(test_dataloader, model_1, model_2, config, method, device):
    model_1.eval()
    model_2.eval()

    idx = 0
    target_scores_m1 = []  
    target_scores_m2 = []  
    for anomaly, data, targets in tqdm(test_dataloader, total=len(test_dataloader)):

        data, targets = data.cuda(device), targets.cuda(device)
        with torch.no_grad():
            # A2Nでは，仕様変更対象:異常
            # N2Aでは，仕様変更対象:正常
            outputs_1 = forward_model(model_1, data, config, method, device)
            # A2Nでは，仕様変更対象:正常
            # N2Aでは，仕様変更対象:異常
            outputs_2 = forward_model(model_2, data, config, method, device)

        s1 = outputs_1.reshape(data.shape[0], -1).max(axis=-1)[0].detach().cpu().numpy()
        s2 = outputs_2.reshape(data.shape[0], -1).max(axis=-1)[0].detach().cpu().numpy()

        for i, ano_type in enumerate(anomaly):
            if "good-" in ano_type or "pseudo_anomaly" in ano_type:
                target_scores_m1.append(s1[i])
                target_scores_m2.append(s2[i])

        idx += 1
    return target_scores_m1, target_scores_m2

def eval_once(config, train_dataloader, test_dataloader, model_1, model_2, args, broken_af, method, device):

    if method == "GLASS":
        target_scores_m1, target_scores_m2 = eval_loop_glass(test_dataloader, model_1, model_2, device)
    elif method == "SimpleNet":
        target_scores_m1, target_scores_m2 = eval_loop_simplenet(test_dataloader, model_1, model_2, device)
    else:
        target_scores_m1, target_scores_m2 = eval_loop(test_dataloader, model_1, model_2, config, method, device)

    results = {}

    # ラベル: モデル1(変更前)=1, モデル2(変更後)=0
    if len(target_scores_m1) > 0:
        if "A2N" in args.scenario:
            y_true_shift = np.concatenate([np.ones(len(target_scores_m1)), np.zeros(len(target_scores_m2))])
            y_score_shift = np.concatenate([target_scores_m1, target_scores_m2])
        elif "N2A" in args.scenario:
            y_true_shift = np.concatenate([np.zeros(len(target_scores_m1)), np.ones(len(target_scores_m2))])
            y_score_shift = np.concatenate([target_scores_m1, target_scores_m2])

        results['Shift_AUROC'] = roc_auc_score(y_true_shift, y_score_shift)

    print("\n--- Specification Change Analysis ---")
    print(f"Shift AUROC (M1 vs M2): {results['Shift_AUROC']:.2%} (Target: 100%)")
    return results

def evaluate(args):
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")
    if args.gpu == -1 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    if "A2N" in args.scenario:
        broken_type_before = "None"
        broken_type_after = _SCENARIO[args.category]
    elif "N2A" in args.scenario:
        broken_type_before = "None"
        broken_type_after = ["pseudo_anomaly"]

    train_dataloader = None
        
    args.result = f"{args.result}/{args.scenario}"
    config = yaml.safe_load(open(args.config, "r"))
    method, _ = os.path.splitext(os.path.basename(args.config))

    # before
    model_1 = build_model(config, method, args.category, device)
    checkpoint_1 = torch.load(f"result/{args.result}/_Normal-exp_{args.category}-{broken_type_before}/best.pt",
                                weights_only=False,
                                )

    model_1.load_state_dict(checkpoint_1["model_state_dict"],strict=False)
    for broken_af in broken_type_after:

        # after
        model_2 = build_model(config, method, args.category, device)
        checkpoint_2 = torch.load(f"result/{args.result}/_{args.scenario}-exp_{args.category}-{broken_af}/best.pt",
                                  weights_only=False,
                                  )
        model_2.load_state_dict(checkpoint_2["model_state_dict"], strict=False)

        test_dataloader = build_test_data_loader(args, config, broken_af)

        model_1.cuda(device)
        model_2.cuda(device)

        result_dict = eval_once(config, train_dataloader, test_dataloader, model_1, model_2, args, broken_af, method, device)
        
        result_path = os.path.join(f"result/{args.result}/_{args.scenario}-exp_{args.category}-{broken_af}/eval_result.txt")
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

    parser.add_argument("--seed", type=int, default=111, help="random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    print(f"category:{args.category}")
    evaluate(args)
