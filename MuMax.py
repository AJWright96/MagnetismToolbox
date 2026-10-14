import pandas as pd
import numpy as np
from scipy.signal import find_peaks

class FMRSimulation():
    def __init__(self):
        pass

    def loadtable(self, paths, skipoverlappingfields=False):
        tables = []
        unique_fields = []
        print("Loading {0} table(s)...".format(len(paths)))
        for path in paths:
            sub_table = pd.read_csv(path, delimiter="\t")
            
            if skipoverlappingfields:
                fields = sub_table['field (oe)'].unique()
                for f in fields: 
                    if f in unique_fields:
                        sub_table = sub_table[sub_table['field (oe)']!=f]
                    else:
                        unique_fields.append(f)
        
        
            tables.append(sub_table)

        self.table = pd.concat(tables, ignore_index=True)

        if skipoverlappingfields:
            self.fields = np.sort(unique_fields)
        else:
            self.fields = np.sort(self.table['field (oe)'].unique())
        print("Tables loaded")
        
    
    def create_pivot(self, values_col, table=None, index='# t (s)', columns='field (oe)', fft=True):
        if table is None:
            table = self.table

        pivot = table.pivot_table(index=index, columns=columns, values=values_col)

        return pivot

    def fft(self, values_col, table=None, window=None, win_mask=None, t_start=0, subtract_t0=True):
        if table is None:
            table = self.table

        table = table[table['# t (s)'] >= t_start]
        p = self.create_pivot(values_col, table=table)
        
        if window=="hanning":
            win_mask = np.hanning(p.values.shape[0])[:,np.newaxis]
        if window=="hanning_half":
            han = np.hanning(p.values.shape[0]*2)[:,np.newaxis]
            win_mask = han[len(han)//2:,:]
        elif window=="drive":
            win_mask = win_mask
        elif window is None:
            win_mask = np.ones(p.values.shape[0])[:,np.newaxis]
        
        if subtract_t0:
            # print()
            p -= p.iloc[0,:]
            # print(np.min(p))

        p_fft = np.fft.rfft(p.values*win_mask, axis=0)

        freqs = np.fft.rfftfreq(len(p.values), d=np.mean(np.diff(p.index)))
        
        # Ignore 0 frequency
        return p_fft[1:], freqs[1:]
    
    def complex_fft(self, values_col1, values_col2, table=None):
        if table is None:
            table = self.table

        p1 = self.create_pivot(values_col1)
        p2 = self.create_pivot(values_col2)

        p_fft = np.fft.fft(p1.values + 1j*p2.values, axis=0)
        p_fft = p_fft[:p_fft.shape[0]//2]

        freqs = np.fft.fftfreq(p1.shape[0], d=np.mean(np.diff(p1.index)))
        freqs = freqs[:p1.shape[0]//2]
        
        return p_fft, freqs

    def find_FMR_peaks(self, data, freqs, height=0.0001, prominence=0.000005):
        film_peak_indices = []
        film_peak_freqs = []
        film_peak_heights_vals = []

        for i in range(data.shape[1]):
            # Find peaks with minimum height and prominence filters
            spectrum = np.abs(data[:, i])
            peaks, properties = find_peaks(spectrum, height=height, prominence=prominence)

            # Sort peaks by height (strongest first)
            film_peak_indices.append(peaks[np.argsort(properties['peak_heights'])[::-1]])
            film_peak_freqs.append(freqs[film_peak_indices[-1]])
            film_peak_heights_vals.append(spectrum[film_peak_indices[-1]])
            
        return film_peak_indices, film_peak_freqs, film_peak_heights_vals             

class OVF():
    def __init__(self):
        pass

    def load_ovf1_binary(self, filename):
        with open(filename, 'rb') as f:
            # Read header lines until the "Begin: Data Binary" line
            header = {}
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith('# Begin: Data Binary'):
                    precision = 4 if '4' in line else 8  # 4 or 8 bytes per float
                    break
                if ':' in line and line.startswith('#'):
                    key, _, val = line[2:].partition(':')
                    header[key.strip()] = val.strip()

            # Read and discard the check value (1234567.0 for float32)
            check = np.frombuffer(f.read(precision), dtype=np.float32 if precision == 4 else np.float64)[0]

            # Read the raw data
            nx = int(header.get('xnodes', 1))
            ny = int(header.get('ynodes', 1))
            nz = int(header.get('znodes', 1))
            n_comp = 3  # vector field (mx, my, mz)

            dtype = np.float32 if precision == 4 else np.float64
            data = np.frombuffer(f.read(nx * ny * nz * n_comp * precision), dtype=dtype)
            data = data.reshape((nz, ny, nx, n_comp))

        self.data = data
        self.header = header

    def load_timesteps(self, ovf_list, zero_time=False):
        for i, f in enumerate(ovf_list):
            # Load file
            d, h = self.load_ovf1_binary(f)

            # If this is the first file, create the empty array to hold all the data
            if i == 0:    
                data = np.zeros((len(ovf_list),) + d.shape)
                ts = np.zeros(len(ovf_list))

            # Load the data into our big array
            data[i,...] = d
            ts[i] = float(h['Desc'].split()[-2])
        
        if zero_time:
            ts -= ts[0]

        return data, ts

    def fft_time(self, data, ts):
        fft = np.fft.fft(data, axis=0)
        fs = np.fft.fftfreq(len(ts), np.average(np.diff(ts*1e9))) # In GHz

        fft = np.fft.fftshift(fft, axes=0)
        fs = np.fft.fftshift(fs)

        return fft, fs

    def fft_myz_time(self, data, ts):
        my = data[...,1]
        mz = data[...,2]
        fft = np.fft.fft(my + 1j*mz, axis=0)
        fs = np.fft.fftfreq(len(ts), np.average(np.diff(ts*1e9))) # In GHz

        fft = np.fft.fftshift(fft, axes=0)
        fs = np.fft.fftshift(fs)

        return fft, fs