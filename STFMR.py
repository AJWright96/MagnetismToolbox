import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm, CenteredNorm, LogNorm
from scipy.signal import savgol_filter
import matplotlib as mpl

class STFMRMeasurement:
    """Class for processing ST-FMR (Spin-Torque Ferromagnetic Resonance) measurement data."""
    
    def __init__(self, filename):
        """Initialize STFRMeasurement with a data file.
        
        Args:
            filename (str): Path to the data file.
        """
        self.filename = filename
        self.data = self.load_data_npz(filename)
        self.set_plot_defaults()

    
    def set_plot_defaults(self):
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['xtick.top'] = True
        mpl.rcParams['ytick.right'] = True
        mpl.rcParams['xtick.minor.visible'] = True
        mpl.rcParams['ytick.minor.visible'] = True

        
    
    def load_data_npz(self, filename):
        """Load ST-FMR data from file (supports pickled .npz format)."""
        try:
            file = np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            print("No such file.")
            pass
        except ValueError:
            print("Cannot unpickle file. Check correct format.")
            pass

        self.x_data = file['x_data']
        self.y_data = file['y_data']
        self.r_data = file['r_data']
        self.theta_data = file['theta_data']

        self.fields = file['fields']
        self.freqs = file['frequencies']

        # Get config information
        # print(dict(file['config']))#['rf_source'])
        cfg = file['config'].item()
        cfg = cfg.strip("{} ")
        config = {}
        for key_pair in cfg.split(","):
            key, value = key_pair.split(":",1)
            config[key.strip('" ' )] = value.strip('" ')

        self.rf_source = config['rf_source']
        self.rf_power = config['rf_power']

        self.start_freq = config['start_freq']
        self.stop_freq = config['stop_freq']
        self.freq_step = config['freq_step']

        self.start_field = config['start_field']
        self.stop_field = config['stop_field']
        self.field_step = config['field_step']
        self.sat_field = config['sat_field']
        self.degauss = config['degauss']

        self.mod_type = config['mod_type']
        self.mod_freq = config['mod_freq']
        self.mod_strength = config['mod_strength']

        self.lockin_sens = config['lockin_sens']
        self.lockin_tc = config['lockin_tc']

        self.sweep_mode = config['sweep_mode']

    def plot_heatmap(self, data, fields, frequencies,
                     zscale='linear', vmin=None, vmax=None, linthresh=None, 
                     vcenter=0, halfrange=None,
                     cmap='RdBu'):
        if zscale=='linear':
            norm = CenteredNorm(vcenter=vcenter, halfrange=halfrange)
        if zscale=='log':
            norm = LogNorm(vmin=vmin, vmax=vmax),
        if zscale=='symlog':
            norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
        
        fig, ax = plt.subplots()
        img = ax.imshow(data, 
                aspect='auto', extent=(fields[0], fields[-1], frequencies[0], frequencies[-1]),
                origin='lower',
                norm=norm, 
                cmap=cmap)
        ax.set_xlabel("Magnetic Field (mT)")
        ax.set_ylabel("Frequency (GHz)")
        plt.colorbar(mappable=img, ax=ax)

        return fig, ax
    
    def plot_linescan(self, data, x_range, x_axis='frequency', y_label='ST-FMR signal (V)',
                        labels=None, legend_kwargs=None,
                        xlim=None, ylim=None,
                        xticks=None, yticks=None,
                        smooth=False, smooth_kwargs=None):
        fig, ax = plt.subplots()

        if len(data.shape) == 1:
            if smooth:
                data = savgol_filter(data, **smooth_kwargs)
            plt.plot(x_range, data, label=labels)
        elif len(data.shape) == 2:
            for i in range(data.shape[1]):
                if smooth:
                    data[:, i] = savgol_filter(data[:, i], **smooth_kwargs)
                plt.plot(x_range, data[:, i], label=(labels[i] if labels is not None else None))

        if x_axis=='frequency':
            ax.set_xlabel("Frequency (GHz)")
        elif x_axis=='field':
            ax.set_xlabel("Magnetic Field (mT)")
        else:
            ax.set_xlabel(x_axis)
        
        ax.set_ylabel(y_label)

        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        if labels is not None:
            ax.legend(**legend_kwargs)

        return fig, ax
        
    def pretty_print_config(self):
        print(f"RF Source: {self.rf_source} at {self.rf_power} dBm")
        print(f"Frequency Sweep: {self.start_freq} GHz to {self.stop_freq} GHz in steps of {self.freq_step} GHz")
        print(f"Field Sweep: {self.start_field} mT to {self.stop_field} mT in steps of {self.field_step} mT")
        print(f"Saturation Field: {self.sat_field} mT, Degauss: {self.degauss}")
        print(f"Modulation: {self.mod_type} at {self.mod_freq} Hz with strength {self.mod_strength} Oe")
        print(f"Lock-in Sensitivity: {self.lockin_sens} V, Time Constant: {self.lockin_tc} s")
        print(f"Sweep Mode: {self.sweep_mode}")