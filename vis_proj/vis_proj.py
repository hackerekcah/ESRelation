from base import VisBase
from helper import get_heat_map
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid

VIS_ROOT = os.path.dirname(os.path.realpath(__file__))


class ProjVis(VisBase):
    def __init__(self,
                 exp,
                 **kwargs):
        super(ProjVis, self).__init__(exp, **kwargs)
        self.save_each = False
        self.show = True
        self.batch_id = 0
        self.target_id = 0
        self.center_tf = (23, 23)
        self.rect_color = 'yellow'

    def center_scan(self):
        half_width = 21
        half_height = 21
        max_t = 250
        max_f = 128

        t_grid = np.linspace(half_width + 2, max_t - half_width - 2, num=6)
        f_grid = np.linspace(half_height + 2, max_f - half_height - 2, num=5)

        center_list = []
        for t in t_grid:
            for f in f_grid:
                center_tf = (int(t), int(f))
                center_list.append(center_tf)
        return center_list

    def fig_structure_grid(self):
        fig = plt.figure(figsize=(7, 1.5))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 5),
                         axes_pad=0.07,
                         share_all=True,
                         cbar_mode='single',
                         label_mode='L')
        im1 = self.fig_spec(ax=grid[0])
        im2 = self.fig_entropy_softmax(ax=grid[1])
        im3 = self.fig_pos_entropy_softmax(ax=grid[2])
        im4 = self.fig_entropy_sparsemax(ax=grid[3])
        im5 = self.fig_pos_entropy_sparsemax(ax=grid[4])

        max_val = im4.get_array().max()
        min_val = im4.get_array().min()
        if max_val > 0.2 and max_val - min_val > 0.1:
            max_val = round(max_val - 0.1, 1)
            min_val = round(min_val, 1)
            plt.colorbar(im4, cax=grid.cbar_axes[0], ticks=[min_val, max_val])
            grid.cbar_axes[0].set_yticklabels([min_val, str(max_val)])
        else:
            plt.colorbar(im3, cax=grid.cbar_axes[0])

        # plt.colorbar(im3, cax=grid.cbar_axes[0])

        fontsz = 12
        grid[0].set_xlabel('(a) spectrogram', fontsize=fontsz, labelpad=6.2)
        grid[1].set_xlabel(r'(b) $\tilde{\mathbf{h}}$', fontsize=fontsz)
        grid[2].set_xlabel(r'(c) $\tilde{\mathbf{h}}^\dag$', fontsize=fontsz)
        grid[3].set_xlabel(r'(d) $\bar{\mathbf{h}}$', fontsize=fontsz)
        grid[4].set_xlabel(r'(e) $\bar{\mathbf{h}}^\dag$', fontsize=fontsz)

        grid[0].get_xaxis().set_ticks([])

        if self.show:
            # fig.suptitle('{}_structure_grid_b{}.png'.format(self.label, str(self.batch_id)))
            plt.show()
        else:
            fig.savefig('{}/{}/{}_structure.png'.format(VIS_ROOT, self.label, self.label))

    def fig_relation_grid(self, suffix=None):
        fig = plt.figure(figsize=(4.8, 1.8))
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(1, 3),
                         axes_pad=0.07,
                         share_all=True,
                         cbar_mode='single',
                         label_mode='L'
                         )
        self.fig_spec_rect(ax=grid[0])
        self.fig_spec_rect(ax=grid[0])
        im1 = self.fig_relation(ax=grid[1])
        im2 = self.fig_pos_relation(ax=grid[2])

        fontsz = 12
        grid[0].set_xlabel('(a) spectrogram', fontsize=fontsz)
        grid[1].set_xlabel(r'(b) $\mathbf{E}_i$', fontsize=fontsz)
        grid[2].set_xlabel(r'(c) $\mathbf{E}_i^{\dag}$', fontsize=fontsz)
        grid[0].get_xaxis().set_ticks([])

        if im1.get_array().max() == 0. and im2.get_array().max() == 0.:
            import matplotlib.colors
            norm = matplotlib.colors.Normalize(vmax=1., vmin=0.)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='jet'), cax=grid.cbar_axes[0])

        elif im1.get_array().max() == 0.:
            plt.colorbar(im2, cax=grid.cbar_axes[0])
        elif im2.get_array().max() == 0.:
            plt.colorbar(im1, cax=grid.cbar_axes[0])
        else:
            plt.colorbar(im1, cax=grid.cbar_axes[0])

        if self.show:
            # fig.suptitle('{}_relation_grid_b{}_{}.png'.format(self.label, self.batch_id, str(self.center_tf)))
            plt.show()
        else:
            if suffix is not None:
                fig.savefig('{}/{}/{}_relation_grid_{}.png'.format(VIS_ROOT,
                                                                   self.label,
                                                                   self.label,
                                                                   str(suffix)))
            else:
                fig.savefig('{}/{}/{}_relation_grid.png'.format(VIS_ROOT,
                                                                self.label,
                                                                self.label))

    def fig_spec(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        self.feed(batch_id=self.batch_id, data_id=0, target_id=self.target_id)

        folder = '{}/{}'.format(VIS_ROOT, self.label)
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.plot_spec(ax=ax)
        if self.save_each:
            fig.savefig('{}/{}/{}_spec.png'.format(VIS_ROOT, self.label, self.label))
        else:
            pass
            # plt.show()

    def fig_spec_rect(self, ax=None):
        if not ax:
            fig, ax = plt.subplots()
        self.feed(batch_id=self.batch_id, data_id=0, target_id=self.target_id)

        folder = '{}/{}'.format(VIS_ROOT, self.label)
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.plot_spec_rect(ax)
        if self.save_each:
            fig.savefig('{}/{}/{}_spec_rect.png'.format(VIS_ROOT, self.label, self.label))
        else:
            pass
            # plt.show()

    def fig_relation(self, ax=None):
        self.reload(exp="esc-folds-rblock",
                    r_structure_type="zero", softmax_type="softmax")
        self.feed(batch_id=self.batch_id, data_id=0, target_id=self.target_id)

        folder = '{}/{}'.format(VIS_ROOT, self.label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not ax:
            fig, ax = plt.subplots()
        im = self.plot_relation_heatmap(ax=ax)
        if self.save_each:
            fig.savefig('{}/{}/{}_relation.png'.format(VIS_ROOT, self.label, self.label))
        else:
            pass
            # plt.show()
        return im

    def fig_pos_relation(self, ax=None):
        self.reload(exp="esc-folds-rblock-pe",
                    r_structure_type="zero", softmax_type="softmax")
        self.feed(batch_id=self.batch_id, data_id=0, target_id=self.target_id)

        folder = '{}/{}'.format(VIS_ROOT, self.label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not ax:
            fig, ax = plt.subplots()
        im = self.plot_relation_heatmap(ax=ax)
        if self.save_each:
            fig.savefig('{}/{}/{}_pos_relation.png'.format(VIS_ROOT, self.label, self.label))
        else:
            pass
            # plt.show()
        return im

    def fig_entropy_softmax(self, ax=None):
        self.reload(exp="esc-folds-rblock",
                    r_structure_type="minus_entropy", softmax_type="softmax")
        self.feed(batch_id=self.batch_id, data_id=0, target_id=self.target_id)

        folder = '{}/{}'.format(VIS_ROOT, self.label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not ax:
            fig, ax = plt.subplots()
        im = self.plot_structure_feat(ax)
        if self.save_each:
            fig.savefig('{}/{}/{}_entropy_softmax.png'.format(VIS_ROOT, self.label, self.label))
        else:
            pass
            # plt.show()

        return im

    def fig_entropy_sparsemax(self, ax=None):
        self.reload(exp="esc-folds-rblock",
                    r_structure_type="minus_entropy", softmax_type="sparsemax")
        self.feed(batch_id=self.batch_id, data_id=0, target_id=self.target_id)

        folder = '{}/{}'.format(VIS_ROOT, self.label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not ax:
            fig, ax = plt.subplots()
        im = self.plot_structure_feat(ax)
        if self.save_each:
            fig.savefig('{}/{}/{}_entropy_sparsemax.png'.format(VIS_ROOT, self.label, self.label))
        else:
            pass
            # plt.show()
        return im

    def fig_pos_entropy_softmax(self, ax=None):
        self.reload(exp="esc-folds-rblock-pe",
                    r_structure_type="minus_entropy", softmax_type="softmax")
        self.feed(batch_id=self.batch_id, data_id=0, target_id=self.target_id)

        folder = '{}/{}'.format(VIS_ROOT, self.label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not ax:
            fig, ax = plt.subplots()
        im = self.plot_structure_feat(ax)
        if self.save_each:
            fig.savefig('{}/{}/{}_pos_entropy_softmax.png'.format(VIS_ROOT, self.label, self.label))
        else:
            pass
            # plt.show()

        return im

    def fig_pos_entropy_sparsemax(self, ax=None):
        self.reload(exp="esc-folds-rblock-pe",
                    r_structure_type="minus_entropy", softmax_type="sparsemax")
        self.feed(batch_id=self.batch_id, data_id=0, target_id=self.target_id)

        folder = '{}/{}'.format(VIS_ROOT, self.label)
        if not os.path.exists(folder):
            os.makedirs(folder)
        if not ax:
            fig, ax = plt.subplots()
        im = self.plot_structure_feat(ax)
        if self.save_each:
            fig.savefig('{}/{}/{}_pos_entropy_sparsemax.png'.format(VIS_ROOT, self.label, self.label))
        else:
            pass
            # plt.show()

        return im

    def plot_spec(self, ax):
        ax.imshow(self.spec, cmap='magma', origin='lower')

    def plot_spec_rect(self, ax):
        ax.imshow(self.spec, cmap='magma', origin='lower')
        self.plot_rect(ax)

    def plot_rect(self, ax, text=None):
        width = 43
        height = 43
        lower_left = (self.center_tf[0] - math.floor(width / 2), self.center_tf[1] - math.floor(height / 2))

        rect = Rectangle(xy=lower_left, width=width, height=height, linewidth=1,
                         edgecolor=self.rect_color, facecolor='none')
        ax.add_patch(rect)
        # ax.scatter(self.center_tf[0], self.center_tf[1], s=10,  marker='x', c=self.rect_color)
        if text == 'p':
            ax.text(self.center_tf[0] - 10, self.center_tf[1] - 8, r'$p$', fontsize=10, color=self.rect_color)
        elif text == 'q':
            ax.text(self.center_tf[0] - 10, self.center_tf[1] - 8, r'$q$', fontsize=10, color=self.rect_color)

    def plot_relation_heatmap(self, ax, fig=None, alpha=1.):
        fsz, tsz = self.spec.shape
        heat_map = get_heat_map(self.spec, nl_map=self.nl_map, center_tf=self.center_tf)

        # (F, T)
        heat_map = F.interpolate(torch.from_numpy(heat_map),
                                 size=(fsz, tsz),
                                 mode='bicubic').squeeze()

        heat_map = heat_map.clamp_(min=0.).numpy()

        # alpha, multiply heat_map by alpha
        im = ax.imshow(heat_map, cmap='jet', alpha=alpha, origin='lower')
        self.plot_rect(ax)

        return im

    def plot_structure_feat(self, ax, fig=None, alpha=1.):

        fsz, tsz = self.spec.shape
        structure_feat = F.interpolate(torch.from_numpy(self.relation_feat),
                                       size=(fsz, tsz),
                                       mode='bicubic').squeeze()

        structure_feat.clamp_(min=0., max=1.)

        structure_feat = structure_feat.numpy()

        # alpha, multiply heat_map by alpha
        im = ax.imshow(structure_feat, cmap='bwr', origin='lower', alpha=alpha)

        return im

    def add_colorbar(self, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(cax=cax)


def plot_relation(vis):
    vis.batch_id = 0

    for l in range(50):
        vis.target_id = l
        for i, center_tf in enumerate(vis.center_scan()):
            vis.center_tf = center_tf
            vis.fig_relation_grid(suffix=i)
        audio_path = "{}/{}/{}.wav".format(VIS_ROOT, vis.label, vis.label)
        vis.save_audio(wav_path=audio_path)


def plot_structure(vis):
    vis.batch_id = 0
    # vis.target_id = 47
    # vis.fig_structure_grid()
    for l in range(50):
        vis.target_id = l
        vis.fig_structure_grid()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    vis = ProjVis(exp="esc-folds-rblock",
                  ckpt_prefix="Run029")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['text.usetex'] = True
    plt.rc('font', family='Times Roman')
    vis.show = False
    plot_relation(vis)
    plot_structure(vis)

    # vis.target_id = 23
    # for i, center_tf in enumerate(vis.center_scan()):
    #     # if i != 6:
    #     #     continue
    #     vis.center_tf = center_tf
    #     vis.fig_relation_grid(suffix=i)
    #     # break



