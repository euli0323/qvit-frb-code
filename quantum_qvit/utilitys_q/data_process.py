'鏁版嵁鐩稿叧'

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# from model_q.q_model import size_image

def per_image_zscore(img_tensor, eps=1e-6):
    mean = img_tensor.mean()
    std = img_tensor.std()
    return (img_tensor - mean) / (std + eps)

def build_transforms(config):
    size_image = config['size_image']
    short_side = min(size_image) if isinstance(size_image, (list, tuple)) else size_image

    train_ops = [
        transforms.Grayscale(1),
        transforms.Resize(short_side),
        transforms.CenterCrop(size_image),
    ]
    translate = config.get('random_translate')
    if translate:
        if isinstance(translate, (list, tuple)):
            translate = tuple(translate)
        else:
            translate = (translate, translate)
        train_ops.append(transforms.RandomAffine(degrees=0, translate=translate))
    train_ops.extend([
        transforms.ToTensor(),
        transforms.Lambda(per_image_zscore),
    ])
    train = transforms.Compose(train_ops)

    val = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(short_side),
        transforms.CenterCrop(size_image),
        transforms.ToTensor(),
        transforms.Lambda(per_image_zscore),
    ])

    return train, val


def build_datasets(data_dir, transform_train, transform_val):
    train_dataset = datasets.ImageFolder(data_dir/'train', transform=transform_train)
    val_dataset = datasets.ImageFolder(data_dir/'val', transform=transform_val)
    test_dataset = datasets.ImageFolder(data_dir/'test', transform=transform_val)
    
    return train_dataset, val_dataset, test_dataset


def build_dataloader(dataset, 
                     batch_size:int, 
                     shuffle: bool):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
