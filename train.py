import os
import torch
import random
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from net.CIDNet import CIDNet
from data.options import option
from measure import metrics
from eval import eval
from data.data import *
from loss.losses import *
from data.scheduler import *
from tqdm import tqdm
from datetime import datetime
from device_utils import empty_cache, resolve_device, tensor_to_pil_image, warn_if_fallback

opt = option().parse_args()
device = torch.device('cpu')


class DummyTrainingDataset(Dataset):
    def __init__(self, crop_size, length):
        self.crop_size = crop_size
        self.length = length

    def __getitem__(self, index):
        low = torch.rand(3, self.crop_size, self.crop_size)
        high = torch.rand(3, self.crop_size, self.crop_size)
        name = f'dry_run_{index:04d}.png'
        return low, high, name, name

    def __len__(self):
        return self.length


class DummyEvalDataset(Dataset):
    def __init__(self, crop_size, length, norm_size=True):
        self.crop_size = crop_size
        self.length = length
        self.norm_size = norm_size

    def __getitem__(self, index):
        image = torch.rand(3, self.crop_size, self.crop_size)
        name = f'dry_run_eval_{index:04d}.png'
        if self.norm_size:
            return image, name
        return image, name, self.crop_size, self.crop_size

    def __len__(self):
        return self.length

def seed_torch():
    seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def train_init():
    global device
    seed_torch()
    device = resolve_device(opt.gpu_mode)
    warn_if_fallback(opt.gpu_mode, device, context='train')
    opt.gpu_mode = (device.type == 'cuda')
    cudnn.benchmark = (device.type == 'cuda')
    if device.type == 'cpu' and os.name == 'nt' and opt.threads > 0:
        print(f'===> Windows CPU mode detected, setting dataloader workers from {opt.threads} to 0 for stability.')
        opt.threads = 0
    print(f'===> Using device: {device}')
    
def train(epoch):
    model.train()
    loss_print = 0
    pic_cnt = 0
    loss_last_10 = 0
    pic_last_10 = 0
    train_len = len(training_data_loader)
    iter = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)
    for batch in tqdm(training_data_loader):
        im1, im2, path1, path2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.to(device)
        im2 = im2.to(device)
        
        # use random gamma function (enhancement curve) to improve generalization
        if opt.gamma:
            gamma = random.randint(opt.start_gamma,opt.end_gamma) / 100.0
            output_rgb = model(im1 ** gamma)  
        else:
            output_rgb = model(im1)  
            
        gt_rgb = im2
        output_hvi = model.HVIT(output_rgb)
        gt_hvi = model.HVIT(gt_rgb)
        loss_hvi = L1_loss(output_hvi, gt_hvi) + D_loss(output_hvi, gt_hvi) + E_loss(output_hvi, gt_hvi) + opt.P_weight * P_loss(output_hvi, gt_hvi)[0]
        loss_rgb = L1_loss(output_rgb, gt_rgb) + D_loss(output_rgb, gt_rgb) + E_loss(output_rgb, gt_rgb) + opt.P_weight * P_loss(output_rgb, gt_rgb)[0]
        loss = loss_rgb + opt.HVI_weight * loss_hvi
        iter += 1

        optimizer.zero_grad()
        loss.backward()
        if opt.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01, norm_type=2)
        optimizer.step()
        
        loss_print = loss_print + loss.item()
        loss_last_10 = loss_last_10 + loss.item()
        pic_cnt += 1
        pic_last_10 += 1
        if iter == train_len:
            print("===> Epoch[{}]: Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                loss_last_10/pic_last_10, optimizer.param_groups[0]['lr']))
            loss_last_10 = 0
            pic_last_10 = 0
            output_img = tensor_to_pil_image(output_rgb[0])
            gt_img = tensor_to_pil_image(gt_rgb[0])
            training_folder = os.path.join(opt.val_folder, 'training')
            os.makedirs(training_folder, exist_ok=True)
            output_img.save(os.path.join(training_folder, 'test.png'))
            gt_img.save(os.path.join(training_folder, 'gt.png'))
    return loss_print, pic_cnt
                

def checkpoint(epoch):
    os.makedirs("./weights/train", exist_ok=True)
    model_out_path = "./weights/train/epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path
    
def load_datasets():
    print(f'===> Loading datasets: {opt.dataset}')
    if opt.dry_run:
        print('===> Dry run mode enabled: using synthetic random tensors instead of reading dataset folders.')
        train_set = DummyTrainingDataset(crop_size=opt.cropSize, length=opt.dry_run_train_samples)
        test_set = DummyEvalDataset(crop_size=opt.cropSize, length=opt.dry_run_eval_samples)
        training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batchSize, shuffle=False)
        testing_data_loader = DataLoader(dataset=test_set, num_workers=0, batch_size=1, shuffle=False)
        return training_data_loader, testing_data_loader

    if opt.dataset == 'lol_v1':
        train_set = get_lol_training_set(opt.data_train_lol_v1,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_v1)
        
    elif opt.dataset == 'lol_blur':
        train_set = get_training_set_blur(opt.data_train_lol_blur,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lol_blur)

    elif opt.dataset == 'lolv2_real':
        train_set = get_lol_v2_training_set(opt.data_train_lolv2_real,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_real)
        
    elif opt.dataset == 'lolv2_syn':
        train_set = get_lol_v2_syn_training_set(opt.data_train_lolv2_syn,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_lolv2_syn)
    
    elif opt.dataset == 'SID':
        train_set = get_SID_training_set(opt.data_train_SID,size=opt.cropSize)
        test_set = get_eval_set(opt.data_val_SID)
        
    elif opt.dataset == 'SICE_mix':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_mix)
        
    elif opt.dataset == 'SICE_grad':
        train_set = get_SICE_training_set(opt.data_train_SICE,size=opt.cropSize)
        test_set = get_SICE_eval_set(opt.data_val_SICE_grad)
        
    elif opt.dataset == 'fivek':
        train_set = get_fivek_training_set(opt.data_train_fivek,size=opt.cropSize)
        test_set = get_fivek_eval_set(opt.data_val_fivek)
    else:
        raise Exception("should choose a dataset")
    
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=opt.shuffle)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)
    return training_data_loader, testing_data_loader

def build_model():
    print('===> Building model ')
    model = CIDNet().to(device)
    if opt.start_epoch > 0:
        pth = f"./weights/train/epoch_{opt.start_epoch}.pth"
        model.load_state_dict(torch.load(pth, map_location=device))
    return model

def make_scheduler():
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    total_epochs = max(1, opt.nEpochs)
    remaining_epochs = max(1, total_epochs - opt.start_epoch)

    # The original scheduler config assumes a long real training run.
    # For CPU smoke tests / dry runs, clamp periods so the scheduler stays valid.
    if remaining_epochs <= 1:
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        return optimizer, scheduler

    effective_warmup = min(opt.warmup_epochs, remaining_epochs - 1)
    use_warmup = opt.start_warmup and effective_warmup > 0

    if opt.cos_restart_cyclic:
        first_period = max(1, (remaining_epochs // 4) - effective_warmup)
        second_period = max(1, remaining_epochs - first_period - effective_warmup)
        if use_warmup:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[first_period, second_period],
                restart_weights=[1,1],
                eta_mins=[0.0002,0.0000001]
            )
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=effective_warmup, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[max(1, remaining_epochs // 4), max(1, remaining_epochs - max(1, remaining_epochs // 4))],
                restart_weights=[1,1],
                eta_mins=[0.0002,0.0000001]
            )
    elif opt.cos_restart:
        cosine_period = max(1, remaining_epochs - effective_warmup)
        if use_warmup:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[cosine_period],
                restart_weights=[1],
                eta_min=1e-7
            )
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=effective_warmup, after_scheduler=scheduler_step)
        else:
            scheduler = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[remaining_epochs],
                restart_weights=[1],
                eta_min=1e-7
            )
    else:
        raise Exception("should choose a scheduler")
    return optimizer,scheduler

def init_loss():
    L1_weight   = opt.L1_weight
    D_weight    = opt.D_weight 
    E_weight    = opt.E_weight 
    P_weight    = 1.0
    
    L1_loss= L1Loss(loss_weight=L1_weight, reduction='mean').to(device)
    D_loss = SSIM(weight=D_weight).to(device)
    E_loss = EdgeLoss(loss_weight=E_weight).to(device)
    P_loss = PerceptualLoss({'conv1_2': 1, 'conv2_2': 1,'conv3_4': 1,'conv4_4': 1}, perceptual_weight = P_weight ,criterion='mse').to(device)
    return L1_loss,P_loss,E_loss,D_loss

if __name__ == '__main__':  
    
    '''
    preparision
    '''
    train_init()
    training_data_loader, testing_data_loader = load_datasets()
    model = build_model()
    optimizer,scheduler = make_scheduler()
    L1_loss,P_loss,E_loss,D_loss = init_loss()
    
    '''
    train
    '''
    psnr = []
    ssim = []
    lpips = []
    start_epoch=0
    if opt.start_epoch > 0:
        start_epoch = opt.start_epoch
    os.makedirs(opt.val_folder, exist_ok=True)
    os.makedirs("./results/training", exist_ok=True)
        
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    with open(f"./results/training/metrics{now}.md", "w") as f:
        f.write("dataset: "+ opt.dataset + "\n")  
        f.write(f"lr: {opt.lr}\n")  
        f.write(f"batch size: {opt.batchSize}\n")  
        f.write(f"crop size: {opt.cropSize}\n")  
        f.write(f"HVI_weight: {opt.HVI_weight}\n")  
        f.write(f"L1_weight: {opt.L1_weight}\n")  
        f.write(f"D_weight: {opt.D_weight}\n")  
        f.write(f"E_weight: {opt.E_weight}\n")  
        f.write(f"P_weight: {opt.P_weight}\n")  
        f.write("| Epochs | PSNR | SSIM | LPIPS |\n")  
        f.write("|----------------------|----------------------|----------------------|----------------------|\n")  
        
    for epoch in range(start_epoch+1, opt.nEpochs + start_epoch + 1):
        epoch_loss, pic_num = train(epoch)
        scheduler.step()
        
        if epoch % opt.snapshots == 0:
            model_out_path = checkpoint(epoch) 
            if opt.skip_eval or opt.dry_run:
                if opt.dry_run:
                    print('===> Dry run mode: skipping validation image export and metrics computation.')
                elif opt.skip_eval:
                    print('===> skip_eval=True: skipping validation and metrics computation.')
                empty_cache(device)
                continue

            norm_size = True

            # LOL three subsets
            if opt.dataset == 'lol_v1':
                output_folder = 'LOLv1/'
                label_dir = opt.data_valgt_lol_v1
            if opt.dataset == 'lolv2_real':
                output_folder = 'LOLv2_real/'
                label_dir = opt.data_valgt_lolv2_real
            if opt.dataset == 'lolv2_syn':
                output_folder = 'LOLv2_syn/'
                label_dir = opt.data_valgt_lolv2_syn
            
            # LOL-blur dataset with low_blur and high_sharp_scaled
            if opt.dataset == 'lol_blur':
                output_folder = 'LOL_blur/'
                label_dir = opt.data_valgt_lol_blur
                
            if opt.dataset == 'SID':
                output_folder = 'SID/'
                label_dir = opt.data_valgt_SID
                npy = True
            if opt.dataset == 'SICE_mix':
                output_folder = 'SICE_mix/'
                label_dir = opt.data_valgt_SICE_mix
                norm_size = False
            if opt.dataset == 'SICE_grad':
                output_folder = 'SICE_grad/'
                label_dir = opt.data_valgt_SICE_grad
                norm_size = False
                
            if opt.dataset == 'fivek':
                output_folder = 'fivek/'
                label_dir = opt.data_valgt_fivek
                norm_size = False

            im_dir = opt.val_folder + output_folder + '*.png'
            is_lol_v1 = (opt.dataset == 'lol_v1')
            is_lolv2_real = (opt.dataset == 'lolv2_real')
            eval(model, testing_data_loader, model_out_path, opt.val_folder+output_folder, 
                 norm_size=norm_size, LOL=is_lol_v1, v2=is_lolv2_real, alpha=0.8, device=device)
            
            avg_psnr, avg_ssim, avg_lpips = metrics(im_dir, label_dir, use_GT_mean=False, device=device)
            print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
            print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))
            print("===> Avg.LPIPS: {:.4f} ".format(avg_lpips))
            psnr.append(avg_psnr)
            ssim.append(avg_ssim)
            lpips.append(avg_lpips)
            print(psnr)
            print(ssim)
            print(lpips)
            with open(f"./results/training/metrics{now}.md", "a") as f:
                f.write(f"| {epoch} | { avg_psnr:.4f} | {avg_ssim:.4f} | {avg_lpips:.4f} |\n")  
        empty_cache(device)
