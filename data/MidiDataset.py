from torch.utils.data import Dataset


class MidiDataset(Dataset):
    def __init__(self):
        super(MidiDataset, self).__init__()

    def ___len__(self):
        return 0

    def __getitem__(self, item):
        return None
