import torch


def save_checkpoint(state_dict, datapath, filename):
    print('Saving..')
    file = f'{datapath}/{filename}.pth.tar'
    torch.save(state_dict, file)
    print(f'Saved checkpoint to {datapath}/{filename}.pth.tar')


def load_checkpoint(datapath, filename, model, optimizer):
    print('Loading..')
    file = f'{datapath}/{filename}.pth.tar'
    checkpoint = torch.load(file)
    try:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    except KeyError as e:
        print('Invalid key {e} in state dict')
    print(f'Loaded checkpoint from {file}')
