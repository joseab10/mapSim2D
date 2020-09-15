import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.lines import Line2D

from map_simulator.disc_states import DiscreteStates as DiSt


class MapColorizer:

    def __init__(self, wm_extent=None, ds_list=None):

        self._wm_extent = None
        self._aspect_ratio = 'lower'

        self._img_origin = 'upper'

        if wm_extent is None:
            wm_extent = [0, 100, 0, 100]

        self.set_wm_extent(wm_extent)

        if ds_list is None:
            ds_list = DiSt.list_all()

        self._cb_orientation = 'horizontal'

        # Discrete State Parameters
        self._ds_list = None

        self._clr_ds = None
        self._cmp_ds = None
        self._tks_ds = None
        self._tlb_ds = None
        self._bnd_ds = None
        self._nrm_ds = None
        self._map_ds = None

        self.set_disc_state_list(ds_list)

        # Color range
        self._v_min = 0
        self._v_max = 1
        
        self._cb_tick_count = 10

        # Continuous Interval Parameters
        self._cmp_ci = None
        self._tks_ci = None
        self._tlb_ci = None
        self._nrm_ci = None
        self._map_ci = None

        self.set_cont_bounds(None, self._v_min, self._v_max, occupancy_map=True)

    def set_wm_extent(self, wm_extent):
        self._wm_extent = wm_extent
        
    def set_cb_orientation(self, orientation):
        if orientation in ['horizontal', 'vertical']:
            self._cb_orientation = orientation
        else:
            self._cb_orientation = 'horizontal'
    
    def set_aspect_ratio(self, aspect_ratio):
        self._aspect_ratio = aspect_ratio
        
    def set_cb_tick_count(self, cb_tick_count):
        if cb_tick_count > 0:
            self._cb_tick_count = cb_tick_count

    def set_disc_state_list(self, ds_list):

        self._ds_list = DiSt.sort_ds_list(ds_list)

        self._clr_ds = DiSt.get_colors(ds_list)    # Color list for discrete states
        self._cmp_ds = mpl.colors.ListedColormap(self._clr_ds, name="cm_ds")  # Colormap for discrete states

        self._tks_ds = DiSt.get_values(ds_list)  # Tick Values for discrete states
        # Nasty fix for single discrete value
        if len(ds_list) == 1:
            self._tks_ds = [self._tks_ds[0] + i - 1 for i in range(3)]

        self._tlb_ds = DiSt.get_labels(ds_list)  # Tick Labels for discrete states
        if len(ds_list) == 1:
            self._tlb_ds = ['', self._tlb_ds[0], '']

        self._bnd_ds = np.array(self._tks_ds) + 0.5
        self._bnd_ds = np.append(min(self._tks_ds) - 0.5, self._bnd_ds)  # Boundaries for the colors in the
        self._nrm_ds = BoundaryNorm(self._tks_ds, len(self._tks_ds))

        # Scalar Mappable for discrete state color bar
        self._map_ds = mpl.cm.ScalarMappable(
            cmap=self._cmp_ds,
            norm=plt.Normalize(vmin=min(self._tks_ds), vmax=max(self._tks_ds))
        )
        self._map_ds._A = []

    def set_cont_bounds(self, img, v_min=0, v_max=1, occupancy_map=True):

        if v_min is not None:
            self._v_min = v_min
        else:
            self._v_min = img.min()

        if v_max is not None:
            self._v_max = v_max
        else:
            self._v_max = img.max()

        if v_min is None and v_max is None:
            self._cmp_ci = mpl.cm.RdYlBu

        else:
            self._cmp_ci = mpl.cm.afmhot_r

        tick_step = float(self._v_max - self._v_min) / self._cb_tick_count
        self._tks_ci = np.arange(self._v_min, self._v_max + tick_step, tick_step)
        self._tlb_ci = list(np.char.mod('%.2f', self._tks_ci))

        self._nrm_ci = Normalize(vmin=self._v_min, vmax=self._v_max)
        self._map_ci = mpl.cm.ScalarMappable(
            cmap=self._cmp_ci,
            norm=self._nrm_ci
        )
        self._map_ci._A = []

        if occupancy_map:
            if v_min == 0:
                self._tlb_ci[0] +=  '\nFree'
            if v_max == 1:
                self._tlb_ci[-1] += '\nOcc'

    @staticmethod
    def _draw_cb(fig, mappable, params, tick_labels=None):
        cb = fig.colorbar(mappable, **params)

        if tick_labels is not None:
            cb.set_ticklabels(tick_labels)

        return cb

    def _draw_cb_disc(self, fig):
        cb_params = {
            'cmap':        self._cmp_ds,
            'ticks':       self._tks_ds,
            'boundaries':  self._bnd_ds,
            'norm':        self._nrm_ds,
            'orientation': self._cb_orientation,
            'spacing':     'uniform',  # 'proportional'
        }

        return self._draw_cb(fig, self._map_ds, cb_params, self._tlb_ds)

    def _draw_cb_cont(self, fig):
        cb_params = {
            'cmap': self._map_ci,
            'ticks': self._tks_ci,
            'label': "Occupancy",
            'orientation': self._cb_orientation,
            'norm': self._nrm_ci,
            'extend': 'neither'
        }

        return self._draw_cb(fig, self._map_ci, cb_params, self._tlb_ci)

    def _imshow_disc_map(self, ax, img):

        params = {
            'cmap': self._cmp_ds,
            'extent': self._wm_extent,
            'vmin': min(self._tks_ds),
            'vmax': max(self._tks_ds),
            'origin': self._img_origin
        }

        return ax.imshow(img, **params)

    def _imshow_cont_map(self, ax, img, v_min=0, v_max=1, occupancy_map=True):

        self.set_cont_bounds(img, v_min=v_min, v_max=v_max, occupancy_map=occupancy_map)

        params = {
            'cmap': self._cmp_ci,
            'extent': self._wm_extent,
            'vmin': self._v_min,
            'vmax': self._v_max,
            'origin': self._img_origin
        }

        return ax.imshow(img, **params)

    def _make_figure(self):
        fig, ax = plt.subplots()

        ax.set_aspect(self._aspect_ratio)

        return fig, ax

    def _draw_plot(self, cont_map, ds_map=None, v_min=0, v_max=1, occupancy_map=True):
        fig, ax = self._make_figure()

        if ds_map is not None:
            self._imshow_disc_map(ax, ds_map)
            self._draw_cb_disc(fig)

        self._imshow_cont_map(ax, cont_map, v_min=v_min, v_max=v_max, occupancy_map=occupancy_map)
        self._draw_cb_cont(fig)

        return fig, ax

    def colorize(self, cont_map, ds_map=None, v_min=0, v_max=1):
        shape = cont_map.shape
        shape = (shape[0], shape[1], 4)
        rgba_img = np.zeros(shape)

        if ds_map is not None:
            ds_map = self._nrm_ds(ds_map)
            rgba_img += self._cmp_ds(ds_map)

        self.set_cont_bounds(cont_map, v_min=v_min, v_max=v_max)
        cont_map = self._nrm_ci(cont_map)
        rgba_img += self._cmp_ci(cont_map)

        return rgba_img

    def plot(self, cont_map, ds_map=None, v_min=0, v_max=1, occupancy_map=True):
        fig, ax = self._draw_plot(cont_map, ds_map, v_min=v_min, v_max=v_max, occupancy_map=occupancy_map)

        fig.show()
    
    def plot_save(self, path, cont_map, ds_map=None, v_min=0, v_max=1, occupancy_map=True):
        self._draw_plot(cont_map, ds_map, v_min=v_min, v_max=v_max, occupancy_map=occupancy_map)
        
        plt.savefig(path, bbox_inches='tight')


# Example and test code
if __name__ == '__main__':

    import numpy.ma as ma
    import os.path

    hits = np.array([[0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 0],
                     [0, 1, 1, 2, 0],
                     [0, 2, 1, 3, 0],
                     [0, 0, 0, 0, 1]])

    visits = np.array([[0, 0, 0, 0, 0],
                       [0, 1, 1, 1, 0],
                       [0, 1, 9, 3, 0],
                       [0, 2, 2, 4, 0],
                       [0, 0, 0, 0, 8]])

    undef_mask = (visits == 0)
    alpha = ma.masked_array(hits, dtype=np.float)
    alpha[undef_mask] = ma.masked

    means = ma.divide(alpha, visits)
    
    means_ds = ma.zeros(means.shape)
    means_ds[undef_mask] = DiSt.UNDEFINED.value
    means_ds[~undef_mask] = ma.masked

    worldmap_extent = [150.4, 183.0, 24.5, 0]
    test_ds_list = [DiSt.UNDEFINED, DiSt.UNIFORM, DiSt.BIMODAL]

    test_v_min = 0
    test_v_max = 10

    test_occ = True

    # Create Colorizer Object
    mean_colorizer = MapColorizer()
    mean_colorizer.set_wm_extent(worldmap_extent)
    mean_colorizer.set_disc_state_list(test_ds_list)
    mean_colorizer.set_cb_tick_count(4)
    mean_colorizer.set_aspect_ratio('equal')
    mean_colorizer.set_cb_orientation('vertical')

    test_path = os.path.expanduser("~/Desktop/colorizer_test.svg")
    mean_colorizer.plot_save(test_path, means, means_ds, v_min=test_v_min, v_max=test_v_max, occupancy_map=test_occ)

    # Plot map
    mean_colorizer.plot(means, means_ds, v_min=test_v_min, v_max=test_v_max, occupancy_map=test_occ)

    rgba = mean_colorizer.colorize(means, means_ds, v_min=test_v_min, v_max=test_v_max)
    plt.figure()
    plt.imshow(rgba)
    plt.show()

    print('end')
