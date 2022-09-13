import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch import nn, optim


class AE(nn.Module):
    '''
    classical Autoencoder for dimensional reduction
    '''

    def __init__(self, dim):
        super(AE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 512)
        self.fc4 = nn.Linear(512, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        z = self.encode(x.view(-1, self.dim))
        return self.decode(z), z


class VAE(nn.Module):
    '''
    Classical Variational Autoencoder for dimensional reduction
    '''

    def __init__(self, dim):
        super(VAE, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.dim))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class PVAE(nn.Module):
    def __init__(self, D,
                 qlayers, qdim,
                 players, pdim,
                 in_dim, zdim=1,
                 outdim=10000,
                 encode_mode='resid',
                 decode_mode='resid',
                 enc_type='linear_lowf',
                 enc_dim=None,
                 domain='fourier',
                 activation=nn.ReLU,
                 petype='concat',
                 pe_alpha=1.0):
        super(PVAE, self).__init__()
        self.D = D
        self.zdim = zdim
        self.in_dim = in_dim
        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, zdim * 2)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(in_dim,
                                          qlayers,  # nlayers
                                          qdim,  # hidden_dim
                                          zdim * 2,  # out_dim
                                          activation)
        elif encode_mode == 'mlp':
            self.encoder = MLP(in_dim,
                               qlayers,
                               qdim,  # hidden_dim
                               zdim * 2,  # out_dim
                               activation)  # in_dim -> hidden_dim
        elif encode_mode == 'cluster':
            self.encoder = clusterEnMLP(in_dim, zdim * 2)
        else:
            raise RuntimeError(
                'Encoder mode {} not recognized'.format(encode_mode))
        self.encode_mode = encode_mode
        self.decode_mode = decode_mode
        self.petype = petype
        self.pe_alpha = pe_alpha
        self.decoder = get_decoder(
            2 + zdim, D, players, pdim, domain, decode_mode, enc_type, outdim, enc_dim, activation, petype)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        z = self.encoder(x)
        return z[:, :self.zdim], z[:, self.zdim:]

    def cat_z(self, coords, z):
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat(
            (coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, z, mask=None):
        '''
        coords: BxNx2 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords, z))

    def forward(self, coords, z):
        return self.decode(coords, z)


class PAE(nn.Module):
    def __init__(self, D,
                 qlayers, qdim,
                 players, pdim,
                 in_dim, zdim=1,
                 outdim=10000,
                 encode_mode='resid',
                 decode_mode='resid',
                 enc_type='linear_lowf',
                 enc_dim=None,
                 domain='fourier',
                 activation=nn.ReLU,
                 petype='concat',
                 pe_alpha=1.0):
        super(PAE, self).__init__()
        self.D = D
        self.zdim = zdim
        self.in_dim = in_dim
        if encode_mode == 'conv':
            self.encoder = ConvEncoder(qdim, zdim)
        elif encode_mode == 'resid':
            self.encoder = ResidLinearMLP(in_dim,
                                          qlayers,  # nlayers
                                          qdim,  # hidden_dim
                                          zdim,  # out_dim
                                          activation)
        elif encode_mode == 'mlp':
            self.encoder = MLP(in_dim,
                               qlayers,
                               qdim,  # hidden_dim
                               zdim,  # out_dim
                               activation)  # in_dim -> hidden_dim
        elif encode_mode == 'cluster':
            self.encoder = clusterEnMLP(in_dim, zdim)
        else:
            raise RuntimeError(
                'Encoder mode {} not recognized'.format(encode_mode))
        self.encode_mode = encode_mode
        self.decode_mode = decode_mode
        self.petype = petype
        self.pe_alpha = pe_alpha
        self.decoder = get_decoder(
            2 + zdim, D, players, pdim, domain, decode_mode, enc_type, outdim, enc_dim, activation, petype, pe_alpha)

    def reparameterize(self, mu, logvar):
        if not self.training:
            return mu
        std = torch.exp(.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        return self.encoder(x)

    def cat_z(self, coords, z):
        '''
        coords: Bx...x3
        z: Bxzdim
        '''
        assert coords.size(0) == z.size(0)
        z = z.view(z.size(0), *([1] * (coords.ndimension() - 2)), self.zdim)
        z = torch.cat(
            (coords, z.expand(*coords.shape[:-1], self.zdim)), dim=-1)
        return z

    def decode(self, coords, z, mask=None):
        '''
        coords: BxNx2 image coordinates
        z: Bxzdim latent coordinate
        '''
        return self.decoder(self.cat_z(coords, z))

    def forward(self, coords, z):
        return self.decode(coords, z)


def get_decoder(in_dim, D, layers, dim, domain, decode_mode, enc_type, outdim, enc_dim=None, activation=nn.ReLU,
                petype='concat', pe_alpha=1.0):
    if enc_type == 'none':
        # TODO
        if domain == 'hartley':
            model = ResidLinearMLP(in_dim, layers, dim, 1, activation)
            ResidLinearMLP.eval_volume = PositionalDecoder.eval_volume  # EW FIXME
        else:
            model = FTSliceDecoder(in_dim, D, layers, dim, activation, petype)
        return model
    else:
        model = PositionalDecoder if domain == 'hartley' else FTPositionalDecoder
        return model(in_dim, D, layers, dim, activation, petype, pe_alpha, decode_mode=decode_mode, enc_type=enc_type,
                     enc_dim=enc_dim, outdim=outdim)


class PositionalDecoder(nn.Module):
    # def __init__(self, in_dim, D,  petype, pe_alpha, enc_type='linear_lowf', enc_dim=None):
    def __init__(self, in_dim, D, nlayers, hidden_dim, activation, petype, pe_alpha, decode_mode='resid', enc_type='linear_lowf', enc_dim=None, outdim=None):
        super(PositionalDecoder, self).__init__()
        assert in_dim >= 2
        self.zdim = in_dim - 2
        self.D = D
        self.D2 = D // 2  # D//2
        self.DD = 2 * (D // 2)  # D//2
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        self.enc_type = enc_type
        self.in_dim = 2 * (self.enc_dim) * 2 + self.zdim
        self.alpha = pe_alpha
        self.petype = petype
        if petype == 'concat':
            type_in_dim = self.in_dim
        elif petype == 'add':
            type_in_dim = self.zdim
        # self.decoder = ResidLinearMLP(type_in_dim, nlayers, hidden_dim, outdim,
        #                               activation) if decode_mode == 'resid' else clusterDeMLP(type_in_dim, outdim)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            # option 1: 2/D to 1
            freqs = self.DD * np.pi * (2. / self.DD) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == 'geom_full':
            # option 2: 2/D to 2pi
            freqs = self.DD * np.pi * (1. / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == 'geom_lowf':
            # option 3: 2/D*2pi to 2pi
            freqs = self.D2 * (1. / self.D2) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == 'geom_nohighf':
            # option 4: 2/D*2pi to 1
            freqs = self.D2 * (2. * np.pi / self.D2) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        else:
            raise RuntimeError(
                'Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 2 x 1
        k = coords[..., 0:2, :] * freqs  # B x 2 x D2
        s = torch.sin(k)  # B x 2 x D2
        c = torch.cos(k)  # B x 2 x D2
        x = torch.cat([s, c], -1)  # B x 2 x D
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 2:, :].squeeze(-1)], -1)
            assert x.shape[-1] == self.in_dim
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 2 x 1
        k = coords[..., 0:2, :] * freqs  # B x 2 x D2
        s = torch.sin(k)  # B x 2 x D2
        c = torch.cos(k)  # B x 2 x D2
        x = torch.cat([s, c], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.in_dim -
                   self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            if self.petype == 'concat':
                x = torch.cat([x, coords[..., 2:, :].squeeze(-1)], -1)
                assert x.shape[-1] == self.in_dim
            elif self.petype == 'add':
                tmp = torch.add(x[:, :self.D], coords[..., 2:, :].squeeze(-1), alpha=self.alpha)
                x = torch.add(tmp, x[:, self.D:], alpha=self.alpha)
        return x

    def forward(self, coords):
        '''Input should be coordinates from [-.5,.5]'''
        assert (coords[..., 0:2].abs() - 0.5 < 1e-4).all()
        # return self.positional_encoding_geom(coords)
        return self.decoder(self.positional_encoding_geom(coords))


    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume
        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        # Note: extent should be 0.5 by default, except when a downsampled
        # volume is generated
        if zval is not None:
            zdim = len(zval)
            z = torch.zeros(D ** 2, zdim, dtype=torch.float32)
            z += torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D, D, D), dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)):
            x = coords + torch.tensor([0, 0, dz])
            if zval is not None:
                x = torch.cat((x, z), dim=-1)
            with torch.no_grad():
                y = self.forward(x)
                y = y.view(D, D).cpu().numpy()
            vol_f[i] = y
        vol_f = vol_f * norm[1] + norm[0]
        # remove last +k freq for inverse FFT
        vol = ihtn_center(vol_f[0:-1, 0:-1, 0:-1])
        return vol


class FTPositionalDecoder(nn.Module):
    # def __init__(self, in_dim, D, nlayers, hidden_dim, activation, petype, pe_alpha, decode_mode='resid',
    #              enc_type='linear_lowf', enc_dim=None, outdim=None):
    def __init__(self, in_dim, D, petype, pe_alpha, enc_type = 'linear_lowf', enc_dim = None):
        super(FTPositionalDecoder, self).__init__()
        assert in_dim >= 2
        self.zdim = in_dim - 2
        self.D = D
        self.D2 = D // 2  # D//2
        self.DD = 2 * (D // 2)  # 2*(D//2)
        self.petype = petype
        self.enc_type = enc_type
        self.enc_dim = self.D2 if enc_dim is None else enc_dim
        # add int for fixing bug of float
        self.in_dim = int(2 * (self.enc_dim) * 2 + self.zdim)
        self.alpha = pe_alpha
        if petype == 'concat':
            type_in_dim = self.in_dim
        elif petype == 'add':
            type_in_dim = self.zdim
        # self.decoder = ResidLinearMLP(type_in_dim, nlayers, hidden_dim, outdim,
        #                               activation) if decode_mode == 'resid' else clusterDeMLP(type_in_dim, outdim)

    def positional_encoding_geom(self, coords):
        '''Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi'''
        freqs = torch.arange(self.enc_dim, dtype=torch.float)
        if self.enc_type == 'geom_ft':
            # option 1: 2/D to 1
            freqs = self.DD * np.pi * (2. / self.DD) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == 'geom_full':
            # option 2: 2/D to 2pi
            freqs = self.DD * np.pi * (1. / self.DD / np.pi) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == 'geom_lowf':
            # option 3: 2/D*2pi to 2pi
            freqs = self.D2 * (1. / self.D2) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == 'geom_nohighf':
            # option 4: 2/D*2pi to 1
            freqs = self.D2 * (2. * np.pi / self.D2) ** (freqs / (self.enc_dim - 1))
        elif self.enc_type == 'linear_lowf':
            return self.positional_encoding_linear(coords)
        elif self.enc_type == 'dummy':
            return self.positional_encoding_dummy(coords)
        else:
            raise RuntimeError(
                'Encoding type {} not recognized'.format(self.enc_type))
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 2 x 1
        k = coords[..., 0:2, :] * freqs  # B x 2 x D2
        s = torch.sin(k)  # B x 2 x D2
        c = torch.cos(k)  # B x 2 x D2
        x = torch.cat([s, c], -1)  # B x 2 x D
        # print(x.shape)
        # print(self.D)
        # print(torch.max(x))
        # print(torch.min(x))
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        # np.save('defaultPE',x.detach().numpy())
        # print(x.detach().numpy())
        if self.zdim > 0:
            if self.petype == 'concat':
                x = torch.cat([x, coords[..., 2:, :].squeeze(-1)], -1)
                assert x.shape[-1] == self.in_dim
            elif self.petype == 'add':
                tmp = torch.add(x[:, :self.D], coords[..., 2:, :].squeeze(-1), alpha=self.alpha)
                x = torch.add(tmp, x[:, self.D:], alpha=self.alpha)
            # print(x)
        return x

    def positional_encoding_linear(self, coords):
        '''Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 2 x 1
        k = coords[..., 0:2, :] * freqs  # B x 2 x D2
        s = torch.sin(k)  # B x 2 x D2
        c = torch.cos(k)  # B x 2 x D2
        x = torch.cat([s, c], -1)  # B x 2 x D
        x = x.view(*coords.shape[:-2], self.in_dim -
                   self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            if self.petype == 'concat':
                x = torch.cat([x, coords[..., 2:, :].squeeze(-1)], -1)
                assert x.shape[-1] == self.in_dim
            elif self.petype == 'add':
                tmp = torch.add(x[:, :self.D], coords[..., 2:, :].squeeze(-1), alpha=self.alpha)
                x = torch.add(tmp, x[:, self.D:], alpha=self.alpha)
        return x

    def positional_encoding_dummy(self, coords):
        '''Dummy to test positional encoding
        Expand coordinates in the Fourier basis, i.e. cos(k*n/N), sin(k*n/N), n=0,...,N//2'''
        freqs = torch.arange(1, self.D2 + 1, dtype=torch.float)
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 2 x 1
        k = coords[..., 0:2, :] * freqs  # B x 2 x D2
        # dummy here
        # s = torch.sin(k)  # B x 2 x D2
        # c = torch.cos(k)  # B x 2 x D2
        s = torch.zeros((k.shape[0], k.shape[1], k.shape[2]))  # B x 2 x D2
        c = torch.zeros((k.shape[0], k.shape[1], k.shape[2]))  # B x 2 x D2
        x = torch.cat([s, c], -1)  # B x 2 x D
        # print(x)
        # np.save('defaultPE',x.detach().numpy())
        x = x.view(*coords.shape[:-2], self.in_dim - self.zdim)  # B x in_dim-zdim
        if self.zdim > 0:
            if self.petype == 'concat':
                x = torch.cat([x, coords[..., 2:, :].squeeze(-1)], -1)
                assert x.shape[-1] == self.in_dim
            elif self.petype == 'add':
                tmp = torch.add(x[:, :self.D], coords[..., 2:, :].squeeze(-1), alpha=self.alpha)
                x = torch.add(tmp, x[:, self.D:], alpha=self.alpha)
        return x

    def forward(self, coords):
        '''Input should be coordinates from [-.5,.5]'''
        assert (coords[..., 0:2].abs() - 0.5 < 1e-4).all()
        return self.positional_encoding_geom(coords)
        # return self.decoder(self.positional_encoding_geom(coords))

    def eval_volume(self, coords, D, extent, norm, zval=None):
        '''
        Evaluate the model on a DxDxD volume
        Inputs:
            coords: lattice coords on the x-y plane (D^2 x 3)
            D: size of lattice
            extent: extent of lattice [-extent, extent]
            norm: data normalization
            zval: value of latent (zdim x 1)
        '''
        assert extent <= 0.5
        if zval is not None:
            zdim = len(zval)
            z = torch.tensor(zval, dtype=torch.float32)

        vol_f = np.zeros((D, D, D), dtype=np.float32)
        assert not self.training
        # evaluate the volume by zslice to avoid memory overflows
        for i, dz in enumerate(np.linspace(-extent, extent, D, endpoint=True, dtype=np.float32)):
            x = coords + torch.tensor([0, 0, dz])
            keep = x.pow(2).sum(dim=1) <= extent ** 2
            x = x[keep]
            if zval is not None:
                x = torch.cat((x, z.expand(x.shape[0], zdim)), dim=-1)
            with torch.no_grad():
                if dz == 0.0:
                    y = self.forward(x)
                else:
                    y = self.decode(x)
                    y = y[..., 0] - y[..., 1]
                slice_ = torch.zeros(D ** 2, device='cpu')
                slice_[keep] = y.cpu()
                slice_ = slice_.view(D, D).numpy()
            vol_f[i] = slice_
        vol_f = vol_f * norm[1] + norm[0]
        # remove last +k freq for inverse FFT
        vol = ihtn_center(vol_f[:-1, :-1, :-1])
        return vol


class ResidLinearMLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(ResidLinearMLP, self).__init__()
        layers = [ResidLinear(in_dim, hidden_dim) if in_dim == hidden_dim else nn.Linear(
            in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(ResidLinear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(ResidLinear(hidden_dim, out_dim) if out_dim ==
                                                          hidden_dim else nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class ResidLinear(nn.Module):
    def __init__(self, nin, nout):
        super(ResidLinear, self).__init__()
        self.linear = nn.Linear(nin, nout)
        # self.linear = nn.utils.weight_norm(nn.Linear(nin, nout))

    def forward(self, x):
        z = self.linear(x) + x
        return z


class MLP(nn.Module):
    def __init__(self, in_dim, nlayers, hidden_dim, out_dim, activation):
        super(MLP, self).__init__()
        layers = [nn.Linear(in_dim, hidden_dim), activation()]
        for n in range(nlayers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation())
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class clusterEnMLP(nn.Module):
    def __init__(self, dim, zdim):
        super(clusterEnMLP, self).__init__()
        self.dim = dim
        self.fc1 = nn.Linear(dim, 512)
        self.fc2 = nn.Linear(512, zdim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return F.relu(self.fc2(h1))

    def forward(self, x):
        return self.encode(x.view(-1, self.dim))


class clusterDeMLP(nn.Module):
    def __init__(self, zdim, dim):
        super(clusterDeMLP, self).__init__()
        self.dim = dim
        self.fc3 = nn.Linear(zdim, 512)
        self.fc4 = nn.Linear(512, dim)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, z):
        return self.decode(z)


class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(ConvEncoder, self).__init__()
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is 1 x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, out_dim, 4, 1, 0, bias=False),
            # state size. out_dims x 1 x 1
        )

    def forward(self, x):
        x = x.view(-1, 1, 64, 64)
        x = self.main(x)
        return x.view(x.size(0), -1)  # flatten


def ihtn_center(V):
    V = np.fft.fftshift(V)
    V = np.fft.fftn(V)
    V = np.fft.fftshift(V)
    V /= np.product(V.shape)
    return V.real - V.imag