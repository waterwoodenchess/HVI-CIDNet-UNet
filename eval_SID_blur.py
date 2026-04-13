import os
import argparse
from torchvision.transforms import Compose, ToTensor
from data.data import *
from torch.utils.data import DataLoader
from net.CIDNet import CIDNet
from device_utils import resolve_device, tensor_to_pil_image, warn_if_fallback


def eval(model, testing_data_loader, model_path, output_folder, device):
    torch.set_grad_enabled(False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print('Evaluation: ', output_folder)
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]
            
        input = input.to(device)
        # print(name)
        
        with torch.no_grad():
            output = model(input)
            
        os.makedirs(output_folder, exist_ok=True)
            
        output = torch.clamp(output,0,1)
        output_img = tensor_to_pil_image(output.squeeze(0))
        output_img.save(output_folder + name[0])
        
    torch.set_grad_enabled(True)
    
if __name__ == '__main__':

    eval_parser = argparse.ArgumentParser(description='Eval')
    eval_parser.add_argument('--SID', action='store_true')
    eval_parser.add_argument('--Blur', action='store_true')
    ep = eval_parser.parse_args()

    device = resolve_device(prefer_gpu=True)
    warn_if_fallback(True, device, context='eval_SID_blur')


    net = CIDNet().to(device)
    if ep.Blur:
        for index in range(1,257):
            test_dir = "./datasets/LOL_blur/test/low_blur/"
            fill_index = str(index).zfill(4)
            now_dir = test_dir + fill_index + "/"
            model_path = "./weights/LOL-Blur.pth"
            blur_folder = "./output/LOL_Blur/"
            if not os.path.exists(blur_folder):          
                os.mkdir(blur_folder)  
            if os.path.exists(now_dir):
                output_folder =  blur_folder + fill_index + "/"
                eval_data = DataLoader(dataset=get_eval_set(now_dir), num_workers=0, batch_size=1, shuffle=False)
                eval(net, eval_data, model_path, output_folder, device)
        
    elif ep.SID:
        for index in range(1,230):
            test_dir = "./datasets/Sony_total_dark/test/short/"
            fill_index = '1' + str(index).zfill(4)
            now_dir = test_dir + fill_index + "/"
            model_path = "./weights/SID.pth"
            SID_folder = "./output/SID/"
            if not os.path.exists(SID_folder):          
                os.mkdir(SID_folder)  
            if os.path.exists(now_dir):
                output_folder =  SID_folder + fill_index + "/"
                eval_data = DataLoader(dataset=get_eval_set(now_dir), num_workers=0, batch_size=1, shuffle=False)
                eval(net, eval_data, model_path, output_folder, device)
        


