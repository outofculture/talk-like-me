import numpy as np
import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 6, kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 2, kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def normalize(x):
    shape = (1,) * (x.ndim - 1) + (2,)
    mean = np.array([x[..., i].mean() for i in (0, 1)]).reshape(shape)
    stdev = np.array([x[..., i].std() for i in (0, 1)]).reshape(shape)

    def denormalize(x):
        return (x * stdev) + mean

    return (x - mean) / stdev, denormalize


def complex_to_polar(x):
    return np.concatenate([np.abs(x)[..., None], np.angle(x)[..., None]], axis=x.ndim)


def polar_to_complex(x):
    return x[..., 0] * np.exp(1j * x[..., 1])


def prepare_data(x):
    polar_spec = [complex_to_polar(audio_to_spectrogram(x1))[None, ...] for x1 in x]
    concat = np.concatenate(polar_spec, axis=0)
    norm, denorm = normalize(concat)
    transposed = norm.transpose(0, 3, 1, 2)

    def unprepare(x):
        untrans = x.transpose(0, 2, 3, 1)
        denormed = denorm(untrans)
        audio = [spectrogram_to_audio(polar_to_complex(spec))[None, ...] for spec in denormed]
        return np.concatenate(audio, axis=0)

    return transposed, unprepare


if __name__ == '__main__':
    from torch.autograd import Variable
    import torch.utils.data
    import pyqtgraph as pg
    from audio import load_digits, sample_rate, audio_to_spectrogram, spectrogram_to_audio, play

    digits, labels = load_digits(sample_rate, random=True)

    spec, unprepare_data = prepare_data(digits)

    spec_tensor = torch.tensor(spec)
    loader = torch.utils.data.DataLoader(spec_tensor, batch_size=16)

    model = Autoencoder().cpu()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-5)

    for epoch in range(10):
        for data in loader:
            img = Variable(data).cpu()

            output = model(img)
            loss = distance(output, img)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss.data)
