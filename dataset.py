import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms
import constants as const
import numpy as np
import PIL

_CLASSNAMES = [
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
]

_SCENARIO = {
    "capsule":["crack","faulty_imprint","poke","scratch"],
    "carpet":["metal_contamination"],
    "grid":["broken","glue","metal_contamination"],
    "hazelnut":["cut"],
    "leather":["color","cut","glue","poke"],
    "pill":["color","crack"],
    "screw":["manipulated_front","scratch_head","scratch_neck","thread_side","thread_top"],
}

# class MVTecDataset(torch.utils.data.Dataset):
#     def __init__(self, root, category, input_size, is_train=True):
        
#         if isinstance(input_size, int):
#             input_size = [input_size, input_size]
        
#         self.image_transform = transforms.Compose(
#             [
#                 transforms.Resize(input_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#             ]
#         )
#         if is_train:
#             self.image_files = glob(
#                 os.path.join(root, category, "train", "good", "*.png")
#             )
#         else:
#             self.image_files = glob(os.path.join(root, category, "test", "*", "*.png"))
#             self.target_transform = transforms.Compose(
#                 [
#                     transforms.Resize(input_size),
#                     transforms.ToTensor(),
#                 ]
#             )

#         self.is_train = is_train

#     def __getitem__(self, index):
#         image_file = self.image_files[index]
#         image = Image.open(image_file).convert("RGB")
#         image = self.image_transform(image)
        
#         if self.is_train:
#             return image
#         else:
#             if os.path.dirname(image_file).endswith("good"):
#                 target = torch.zeros([1, image.shape[-2], image.shape[-1]])
#             else:
#                 target = Image.open(
#                     image_file.replace("/test/", "/ground_truth/").replace(
#                         ".png", "_mask.png"
#                     )
#                 ).convert("L")
#                 target = self.target_transform(target)
#             return image, target

#     def __len__(self):
#         return len(self.image_files)



class MVTecDataset_Scenario(torch.utils.data.Dataset):
    def __init__(self, root, category, scenario, broken_type, input_size, use_noise=False, is_train=True):
        
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.scenario = scenario
        self.category = category

        self.use_noise = use_noise
        if use_noise:
            from model.RDplus2.dataset.noise import Simplex_CLASS
            self.simplexNoise = Simplex_CLASS()
        
        if self.scenario == "A2N":
            # assert broken_type in _SCENARIO[category] is not None, "broken_type Error"
            self.broken_type = broken_type
        elif self.scenario == "N2A":
            self.broken_type = "pseudo_anomaly"
        else:
            self.broken_type = None

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        #正常に変更された欠損タイプは"good-"を付ける
        if is_train:
            self.image_files = sorted(glob(os.path.join(root, category, "train", "good", "*.png")))
            self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

            if self.scenario == "A2N":
                #対象欠損タイプの上半分を学習にし、正常として扱う
                target_broken_files = sorted(glob(os.path.join(root, category, "test", self.broken_type, "*.png")))
                target_anomaly_type = ["good-" + path.split("/")[-2] for path in target_broken_files]
                self.image_files = self.image_files + target_broken_files[:len(target_broken_files)//2]
                self.anomaly_type = self.anomaly_type + target_anomaly_type[:len(target_anomaly_type)//2]

            if self.scenario == "N2A":
                # N2AのTrainの場合、学習データの疑似異常は省く
                self.image_files = [img_path for img_path in self.image_files if not "pseudo_" in img_path]
                self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

        else:
            self.image_files = sorted(glob(os.path.join(root, category, "test", "*", "*.png")))
            self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

            if self.scenario == "A2N":
                #対象欠損タイプの下半分をテストにし、正常として扱う
                target_broken_files = sorted(glob(os.path.join(root, category, "test", self.broken_type, "*.png")))

                target_broken_files = target_broken_files[:len(target_broken_files)//2]
                image_files_temp = []
                anomaly_type_temp = []
                for img_file, ano_type in zip(self.image_files, self.anomaly_type):
                    if img_file in target_broken_files:
                        continue
                    if ano_type == self.broken_type:
                        ano_type = f"good-{ano_type}"
                    image_files_temp.append(img_file)
                    anomaly_type_temp.append(ano_type)

                self.image_files = image_files_temp
                self.anomaly_type = anomaly_type_temp

            if self.scenario == "N2A":
                # N2AのTestの場合、疑似異常を異常として扱う。何も変更しなくてよい。
                pass

            if self.scenario == "Normal":
                #N2A-NormalのTestの場合、疑似異常を正常として扱う。
                self.anomaly_type = [f"good-{anomaly}" if anomaly == "pseudo_anomaly" else anomaly for anomaly in self.anomaly_type]

            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )

        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        anomaly = self.anomaly_type[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            if self.use_noise:
                C, H, W = image.shape
                h_noise = np.random.randint(10, H // 8)
                w_noise = np.random.randint(10, W // 8)
                start_h = np.random.randint(0, H - h_noise)
                start_w = np.random.randint(0, W - w_noise)

                simplex_noise = torch.tensor(
                    self.simplexNoise.rand_3d_octaves((C, h_noise, w_noise), 6, 0.6),
                    dtype=image.dtype, device=image.device
                )

                # ノイズ用 Tensor
                noise = torch.zeros_like(image)
                noise[:, start_h:start_h+h_noise, start_w:start_w+w_noise] = 0.2 * simplex_noise
                
                img_noise = image + noise
                
                return anomaly, image, img_noise
            return anomaly, image
        else:
            if "good" in anomaly:
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                ).convert("L")
                target = self.target_transform(target)
            return anomaly, image, target

    def __len__(self):
        return len(self.image_files)
    

class MVTecDataset_Scenario_SimpleNet(torch.utils.data.Dataset):
    def __init__(self, root, category, scenario, broken_type, config, is_train=True):
        input_size = config["data"]["input_size_train"]
        input_size = input_size[1:]
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.scenario = scenario
        self.category = category
        
        if self.scenario == "A2N":
            # assert broken_type in _SCENARIO[category] is not None, "broken_type Error"
            self.broken_type = broken_type
        elif self.scenario == "N2A":
            self.broken_type = "pseudo_anomaly"
        else:
            self.broken_type = None

        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                # transforms.ColorJitter(0.0, 0.0, 0.0),
                # transforms.RandomHorizontalFlip(0.0),
                # transforms.RandomVerticalFlip(0.0),
                # transforms.RandomGrayscale(0.0),
                # transforms.RandomAffine(0, 
                #                         translate=(0, 0),
                #                         scale=(1.0-0, 1.0+0),
                #                         interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        #正常に変更された欠損タイプは"good-"を付ける
        if is_train:
            self.image_files = sorted(glob(os.path.join(root, category, "train", "good", "*.png")))
            self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

            if self.scenario == "A2N":
                #対象欠損タイプの上半分を学習にし、正常として扱う
                target_broken_files = sorted(glob(os.path.join(root, category, "test", self.broken_type, "*.png")))
                target_anomaly_type = ["good-" + path.split("/")[-2] for path in target_broken_files]
                self.image_files = self.image_files + target_broken_files[:len(target_broken_files)//2]
                self.anomaly_type = self.anomaly_type + target_anomaly_type[:len(target_anomaly_type)//2]

            if self.scenario == "N2A":
                # N2AのTrainの場合、学習データの疑似異常は省く
                self.image_files = [img_path for img_path in self.image_files if not "pseudo_" in img_path]
                self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

        else:
            self.image_files = sorted(glob(os.path.join(root, category, "test", "*", "*.png")))
            self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

            if self.scenario == "A2N":
                #対象欠損タイプの下半分をテストにし、正常として扱う
                target_broken_files = sorted(glob(os.path.join(root, category, "test", self.broken_type, "*.png")))

                target_broken_files = target_broken_files[:len(target_broken_files)//2]
                image_files_temp = []
                anomaly_type_temp = []
                for img_file, ano_type in zip(self.image_files, self.anomaly_type):
                    if img_file in target_broken_files:
                        continue
                    if ano_type == self.broken_type:
                        ano_type = f"good-{ano_type}"
                    image_files_temp.append(img_file)
                    anomaly_type_temp.append(ano_type)

                self.image_files = image_files_temp
                self.anomaly_type = anomaly_type_temp

            if self.scenario == "N2A":
                # N2AのTestの場合、疑似異常を異常として扱う。何も変更しなくてよい。
                pass

            if self.scenario == "Normal":
                #N2A-NormalのTestの場合、疑似異常を正常として扱う。
                self.anomaly_type = [f"good-{anomaly}" if anomaly == "pseudo_anomaly" else anomaly for anomaly in self.anomaly_type]

            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                ]
            )

        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        anomaly = self.anomaly_type[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            return {
                "image": image,
                # "mask": mask,
                # "classname": classname,
                "anomaly": anomaly,
                "is_anomaly": int(anomaly != "good"),
                # "image_name": "/".join(image_path.split("/")[-4:]),
                # "image_path": image_path,
            }
        else:
            if "good" in anomaly:
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                ).convert("L")
                target = self.target_transform(target)
            return anomaly, image, target

    def __len__(self):
        return len(self.image_files)

class MVTecDataset_Scenario_GLASS(torch.utils.data.Dataset):
    def __init__(self, root, category, scenario, broken_type, config, is_train=True):
        from model.GLASS.perlin import perlin_mask
        input_size = config["data"]["input_size_train"]
        input_size = input_size[1:]
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.scenario = scenario
        self.category = category
        self.root = root
        self.input_size = input_size
        self.config = config

        if self.scenario == "A2N":
            # assert broken_type in _SCENARIO[category] is not None, "broken_type Error"
            self.broken_type = broken_type
        elif self.scenario == "N2A":
            self.broken_type = "pseudo_anomaly"
        else:
            self.broken_type = None

        #正常に変更された欠損タイプは"good-"を付ける
        if is_train:
            self.image_files = sorted(glob(os.path.join(root, category, "train", "good", "*.png")))
            self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

            if self.scenario == "A2N":
                #対象欠損タイプの上半分を学習にし、正常として扱う
                target_broken_files = sorted(glob(os.path.join(root, category, "test", self.broken_type, "*.png")))
                target_anomaly_type = ["good-" + path.split("/")[-2] for path in target_broken_files]
                self.image_files = self.image_files + target_broken_files[:len(target_broken_files)//2]
                self.anomaly_type = self.anomaly_type + target_anomaly_type[:len(target_anomaly_type)//2]

            if self.scenario == "N2A":
                # N2AのTrainの場合、学習データの疑似異常は省く
                self.image_files = [img_path for img_path in self.image_files if not "pseudo_" in img_path]
                self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

        else:
            self.image_files = sorted(glob(os.path.join(root, category, "test", "*", "*.png")))
            self.anomaly_type = [path.split("/")[-2] for path in self.image_files]

            if self.scenario == "A2N":
                #対象欠損タイプの下半分をテストにし、正常として扱う
                target_broken_files = sorted(glob(os.path.join(root, category, "test", self.broken_type, "*.png")))

                target_broken_files = target_broken_files[:len(target_broken_files)//2]
                image_files_temp = []
                anomaly_type_temp = []
                for img_file, ano_type in zip(self.image_files, self.anomaly_type):
                    if img_file in target_broken_files:
                        continue
                    if ano_type == self.broken_type:
                        ano_type = f"good-{ano_type}"
                    image_files_temp.append(img_file)
                    anomaly_type_temp.append(ano_type)

                self.image_files = image_files_temp
                self.anomaly_type = anomaly_type_temp

            if self.scenario == "N2A":
                # N2AのTestの場合、疑似異常を異常として扱う。何も変更しなくてよい。
                pass

            if self.scenario == "Normal":
                #N2A-NormalのTestの場合、疑似異常を正常として扱う。
                self.anomaly_type = [f"good-{anomaly}" if anomaly == "pseudo_anomaly" else anomaly for anomaly in self.anomaly_type]


        self.target_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
            ]
        )
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.is_train = is_train


        self.rand_aug = config["data"]["rand_aug"]
        if config["data"]["fg"] == 1:  # with foreground mask
            self.class_fg = 1
        else:  # without foreground mask
            self.class_fg = 0
        self.perlin_mask = perlin_mask

        self.anomaly_source_path = config["data"]["anomaly_source_path"]

    def extract_classname(self, image_file, classnames=_CLASSNAMES):
        for cname in classnames:
            if cname in image_file:
                return cname
        return None  # 見つからなければ None
    
    def rand_augmenter(self):
        list_aug = [
            transforms.ColorJitter(contrast=(0.8, 1.2)),
            transforms.ColorJitter(brightness=(0.8, 1.2)),
            transforms.ColorJitter(saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomVerticalFlip(p=1),
            transforms.RandomGrayscale(p=1),
            transforms.RandomAutocontrast(p=1),
            transforms.RandomEqualize(p=1),
            transforms.RandomAffine(degrees=(-45, 45)),
        ]
        aug_idx = np.random.choice(np.arange(len(list_aug)), 3, replace=False)

        transform_aug = [
            transforms.Resize(self.input_size),
            list_aug[aug_idx[0]],
            list_aug[aug_idx[1]],
            list_aug[aug_idx[2]],
            # transforms.CenterCrop(self.imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        transform_aug = transforms.Compose(transform_aug)
        return transform_aug
    
    def get_image_data(self, source, classname, split):
        """
        クラスごと・異常タイプごとの画像パス辞書を作成して返す
        """
        imgpaths_per_class = {}
        classpath = os.path.join(self.root, classname, split)
        anomaly_types = os.listdir(classpath)

        imgpaths_per_class[classname] = {}

        for anomaly in anomaly_types:
            anomaly_path = os.path.join(classpath, anomaly)
            anomaly_files = sorted(os.listdir(anomaly_path))
            imgpaths_per_class[classname][anomaly] = [
                os.path.join(anomaly_path, x) for x in anomaly_files
            ]

        return imgpaths_per_class


    def __getitem__(self, index):
        image_file = self.image_files[index]
        anomaly = self.anomaly_type[index]
        classname = self.extract_classname(image_file)
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            self.anomaly_source_paths = sorted(1 * glob(self.anomaly_source_path + "/*/*.jpg") +
                                           0 * list(next(iter(self.get_image_data(image_file, classname, "train").values())).values())[0])
            if self.rand_aug:
                aug = PIL.Image.open(np.random.choice(self.anomaly_source_paths)).convert("RGB")
                if self.rand_aug:
                    transform_aug_train = self.rand_augmenter()
                    aug = transform_aug_train(aug)
                else:
                    aug = self.transform_img(aug)
                    
            if self.class_fg:
                fgmask_path = image_file.split(classname)[0] + 'fg_mask/' + classname + '/' + os.path.split(image_file)[-1]
                mask_fg = PIL.Image.open(fgmask_path)
                mask_fg = torch.ceil(self.target_transform(mask_fg)[0])

            mask_all = self.perlin_mask(image.shape, image.shape[-1] // 8, 0, 6, mask_fg, 1)
            mask_s = torch.from_numpy(mask_all[0])
            mask_l = torch.from_numpy(mask_all[1])

            beta = np.random.normal(loc=self.config["data"]["mean"], scale=self.config["data"]["std"])
            beta = np.clip(beta, .2, .8)
            aug_image = image * (1 - mask_l) + (1 - beta) * aug * mask_l + beta * image * mask_l
            return {
                "image": image,
                "aug": aug_image,
                "mask_s": mask_s,
                # "mask_gt": mask_gt,
                "is_anomaly": int(anomaly != "good"),
                "image_path": image_file,
            }
        else:
            if "good" in anomaly:
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                target = Image.open(
                    image_file.replace("/test/", "/ground_truth/").replace(
                        ".png", "_mask.png"
                    )
                ).convert("L")
                target = self.target_transform(target)
            return anomaly, image, target

    def __len__(self):
        return len(self.image_files)
