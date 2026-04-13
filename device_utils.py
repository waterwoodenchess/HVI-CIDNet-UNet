import torch
from PIL import Image


def resolve_device(prefer_gpu=True):
    use_cuda = prefer_gpu and torch.cuda.is_available()
    return torch.device('cuda' if use_cuda else 'cpu')


def warn_if_fallback(prefer_gpu, device, context=''):
    if prefer_gpu and device.type != 'cuda':
        prefix = f'{context}: ' if context else ''
        print(f'===> {prefix}CUDA is not available, falling back to CPU.')


def empty_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()


def tensor_to_pil_image(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() != 3:
        raise ValueError(f'Expected a 3D CHW tensor, got shape {tuple(tensor.shape)}')

    channels, height, width = tensor.shape
    byte_tensor = tensor.mul(255).round().to(torch.uint8)

    if channels == 1:
        image_bytes = bytes(byte_tensor.contiguous().view(-1).tolist())
        return Image.frombytes('L', (width, height), image_bytes)

    if channels == 3:
        image_bytes = bytes(byte_tensor.permute(1, 2, 0).contiguous().view(-1).tolist())
        return Image.frombytes('RGB', (width, height), image_bytes)

    raise ValueError(f'Unsupported number of channels: {channels}')


def pil_to_float_tensor(image):
    if not isinstance(image, Image.Image):
        raise TypeError(f'Expected a PIL Image, got {type(image)}')

    image = image.convert('RGB')
    width, height = image.size
    byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
    byte_tensor = byte_tensor.view(height, width, 3)
    return byte_tensor.permute(2, 0, 1).float().div(255.0)
