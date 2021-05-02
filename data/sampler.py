import torch
import torch.utils.data
import random


class BalancedSampler(torch.utils.data.sampler.Sampler):
    """
    Do not use this in test loader, because it oversamples the classes with less samples.
    https://github.com/galatolofederico/pytorch-balanced-batch
    """
    def __init__(self, dataset, labels, shuffle=True):
        """
        :param dataset: Dataset to be sampled, only use len(dataset)
        :param labels: list of int, or torch tensor
        :param shuffle: reshuffle after each epoch
        """
        super(BalancedSampler, self).__init__(dataset)
        if torch.is_tensor(labels):
            self.labels = labels.numpy()
        else:
            self.labels = labels
        self.shuffle = shuffle
        # mapping label to list of sample index
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = int(self.labels[idx])
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            # max number of samples per class
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                # each class extends to have same number of samples (to the classes with max nb of samples)
                # total samples are increased if use unbalanced dataset.
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        # label index
        self.currentkey = 0
        # pointer for each queue of sample indices
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        # shuffle before each epoch
        if self.shuffle:
            for label in self.dataset.keys():
                random.shuffle(self.dataset[label])
        # each time, return one sample index of a class.
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        # reset pointers after iterating all the samples
        self.indices = [-1] * len(self.keys)

    def __len__(self):
        return self.balanced_max * len(self.keys)


if __name__ == '__main__':

    import torch

    epochs = 3
    size = 20
    features = 5
    classes_prob = torch.tensor([0.1, 0.4, 0.5])

    dataset_X = torch.randn(size, features)
    dataset_Y = torch.distributions.categorical.Categorical(classes_prob.repeat(size, 1)).sample()

    dataset = torch.utils.data.TensorDataset(dataset_X, dataset_Y)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               sampler=BalancedBatchSampler(dataset, dataset_Y, shuffle=True),
                                               batch_size=6,
                                               drop_last=True)

    print(len(train_loader))

    for epoch in range(0, epochs):
        for batch_x, batch_y in train_loader:
            print(len(batch_x))
            # print("epoch: %d labels: %s\ninputs: %s\n" % (epoch, batch_y, batch_x))