import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.stats import linregress
import numpy as np



class MOKEMeasurement:
    """Class for processing MOKE (Magneto-Optic Kerr Effect) measurement data, taken on NanoMOKE system."""
    
    def __init__(self, filename, default_plot_style=True):
        """Initialize MOKEMeasurement with a data file.
        
        Args:
            filename (str): Path to the MOKE data file (.mok or .pro format)
        """
        self.filename = filename
        self.data = self.load_data(filename)

        if default_plot_style:
            self.set_plot_defaults()
    
    
    def set_plot_defaults(self):
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        mpl.rcParams['xtick.top'] = True
        mpl.rcParams['ytick.right'] = True
        mpl.rcParams['xtick.minor.visible'] = True
        mpl.rcParams['ytick.minor.visible'] = True

    def load_data(self, filename):
        """Load MOKE data from file (supports .mok and .pro formats).
        
        Args:
            filename (str): Path to the data file
            
        Returns:
            pd.DataFrame: Loaded data with standardized column names
        """
        if filename.endswith(".mok"):
            # Load .mok files with tab delimiter, skip first 6 rows of header
            df = pd.read_csv(filename, delimiter="\t", skiprows=6)
            # Standardize column names for consistency
            df.rename({"Hx(Oe)": "Hx (Oe)", "Long. Kerr(mV)": "Longitudinal Kerr (mV)"}, axis=1, inplace=True)
            return df
        elif filename.endswith(".pro"):
            # Load .pro files with tab delimiter, no header rows to skip
            return pd.read_csv(filename, delimiter="\t", skiprows=0)
    
    def autoprocess(self, remove_slope=True, slope_gradient=None, center_field=True, remove_ellipticity=True, ellipticity=0):
        """Automatically process MOKE data with normalization, slope removal, and ellipticity correction.
        
        Args:
            remove_slope (bool): Whether to remove linear slope from data (default: True)
            slope_gradient (float): Manually specified gradient for slope removal (default: None for auto-detection)
            remove_ellipticity (bool): Whether to apply ellipticity correction (default: True)
            ellipticity (float): Ellipticity parameter for correction (default: 0)
        """
        # Initial normalization to [-1, 1] range
        self.normalise(inplace=True)

        # Remove linear slope if requested
        if remove_slope:
            if slope_gradient:
                self.removeslope(gradient=slope_gradient, inplace=True)
            else:
                self.removeslope(inplace=True)

        if center_field:
            self.center_zerofield(inplace=True)
        
        # Apply ellipticity correction if requested
        if remove_ellipticity:
            self.removeellipticity(ellipticity=ellipticity, inplace=True)
        
        # Final normalization
        self.normalise(inplace=True)

        
        
    def normalise(self, y=None, inplace=False):
        """Normalize y data to [-1, 1] range centered around zero.
        
        Args:
            y (array-like): Data to normalize (default: None uses Longitudinal Kerr column)
            
        Returns:
            array: Normalized data
        """
        if y is None:
            y = np.array(self.data["Longitudinal Kerr (mV)"])
        # Center y data around zero and scale to [-1, 1]
        mid = (np.max(y) + np.min(y))/2
        norm_data = (y - mid)/np.max(y - mid)
        if inplace:
            self.data["Longitudinal Kerr (mV)"] = norm_data
        return norm_data
    
    def center_zerofield(self, x=None, y=None, inplace=False):
        """Center the hysteresis loop around zero magnetic field.
        
        Args:
            data (array-like): Data to process (default: None uses Longitudinal Kerr column)
            
        Returns:
            array: Data centered around zero field
        """
        if x is None: 
            x = np.array(self.data["Hx (Oe)"])
        if y is None:
            y = np.array(self.data["Longitudinal Kerr (mV)"])
        # Find index where magnetisation crosses zero
        forward = np.array(y[:len(y)//2])
        reverse = np.array(y[len(y)//2:])
        i_for =  np.argmin(np.abs(forward))
        i_rev = np.argmin(np.abs(reverse))

        H_res_av = (x[i_for] + x[len(y)//2 + i_rev]) / 2

        if inplace:
            self.data["Hx (Oe)"] = x - H_res_av
                                  
        return x - H_res_av

    def removeslope(self, x=None, y=None, gradient=None, inplace=False):
        """Remove linear slope from data using automatic gradient detection.
        
        Detects the linear component by identifying regions with low curvature
        (low second derivative) and fitting a line through those regions separately
        for forward and reverse scan directions.
        
        Args:
            data (array-like): Data to process (default: None uses self.data)
            gradient (float): Manually specified slope to remove (default: None for auto-detection)
            
        Returns:
            array: Data with linear slope removed
        """
        if x is None: 
            x = self.data["Hx (Oe)"]
        if y is None:
            y = self.data["Longitudinal Kerr (mV)"]

        if gradient == 0:
            return y
        
        if gradient is None:
            # Automatic gradient detection using second derivative analysis
            threshold = 3e-4  # Threshold for identifying low-curvature regions
            forward = np.array(y[:len(y)//2])
            reverse = np.array(y[len(y)//2:])

            # Compute second derivative using Savitzky-Golay filter
            forward_d2y = savgol_filter(forward, 30, 2, deriv=2)
            reverse_d2y = savgol_filter(reverse, 30, 2, deriv=2)

            def find_first_above_threshold(data, threshold):
                """Find first index where absolute value exceeds threshold."""
                for i, x in enumerate(data):
                    if abs(x) > threshold:
                        break
                return i
            
            # Find regions with low curvature (above threshold indicates high curvature/features)
            i_f = find_first_above_threshold(forward_d2y, threshold)
            i_r = find_first_above_threshold(reverse_d2y, threshold)


            # Fit lines through low-curvature regions
            slope_forward = linregress(x[:i_f], forward[:i_f]).slope
            slope_reverse = linregress(x[len(y)//2: len(y)//2 + i_r], reverse[:i_r]).slope

            # Average the slopes from forward and reverse directions
            gradient = (slope_forward + slope_reverse) / 2

        # Subtract linear component from data
        numArray = np.zeros(len(y))
        for i in range(len(numArray)):
            numArray[i] = y[i] - gradient * x[i]

        if inplace:
            self.data["Longitudinal Kerr (mV)"] = numArray
        return numArray

    def removeellipticity(self, x=None, y=None, ellipticity=0, inplace=False):
        """Remove ellipticity-induced distortion using harmonic decomposition.
        
        Ellipticity causes a sinusoidal modulation in the signal. This method
        decomposes the field into sine and cosine components to determine the
        phase and amplitude of the ellipticity distortion, then subtracts it.
        
        Args:
            x (array): Magnetic field data (horizontal axis)
            y (array): Kerr signal data (to be corrected)
            ellipticity (float): Amplitude parameter for ellipticity correction
            
        Returns:
            array: Data with ellipticity correction applied
        """
        if x is None:
            x = self.data["Hx (Oe)"]
        if y is None:
            y = self.data["Longitudinal Kerr (mV)"]
        if ellipticity == 0:
            return y

        numArray = np.zeros(len(y))
        num1 = 0
        num2 = 0
        
        # Decompose into sine and cosine components across the full cycle
        for i in range(len(numArray)):
            num1 += x[i] * np.sin(i/len(numArray) * 2 * np.pi)
            num2 += x[i] * np.cos(i/len(numArray) * 2 * np.pi)
        
        # Calculate harmonic amplitudes
        y1 = 2 * num1/len(numArray)
        x1 = 2 * num2/len(numArray)

        if (y1 != 0) or (x1 != 0):
            # Determine phase of ellipticity distortion
            num3 = np.arctan2(y1, x1)
            # Subtract ellipticity correction term from signal
            for j in range(len(numArray)):
                numArray[j] = y[j] - ellipticity / 2 *np.sin(j / len(numArray) * 2 * np.pi - num3)
        else:
            # No correction needed if harmonic components are zero
            for j in range(len(numArray)):
                numArray[j] = y[j]
        
        if inplace:
            self.data["Longitudinal Kerr (mV)"] = numArray
        
        return numArray

    def plot_loop(self, df, xlim=None, ylim=None, xticks=None, yticks=None,
                  split_loops=False, savefig=False, filename=None):
        """Plot MOKE hysteresis loop (Kerr signal vs. magnetic field).
        
        Args:
            df (pd.DataFrame): Data frame containing 'Hx (Oe)' and 'Longitudinal Kerr (mV)' columns
            filename (str): Optional path to save the figure (default: None, displays only)
        """
        self.fig, self.ax = plt.subplots(1,1,figsize=(5,4))
        # Convert field from Oersteds to millitesla (divide by 10) and plot
        if split_loops:
            colors = mpl.colormaps['tab20c'].colors
            self.ax.plot(df["Hx (Oe)"][:len(df)//2]/10, df["Longitudinal Kerr (mV)"][:len(df)//2], label="Forward", color=colors[0])
            self.ax.plot(df["Hx (Oe)"][len(df)//2:]/10, df["Longitudinal Kerr (mV)"][len(df)//2:], label="Reverse", color=colors[2])
            self.ax.legend()
        else:
            self.ax.plot(df["Hx (Oe)"]/10, df["Longitudinal Kerr (mV)"])
        
        
        self.ax.set_xlabel(r"$\mu_0 H$ (mT)")
        self.ax.set_ylabel("$M/M_s$")

        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

        if filename is not None:
            self.fig.savefig(filename)
        
        if xticks is not None:
            self.ax.set_xticks(xticks)
        if yticks is not None:
            self.ax.set_yticks(yticks)

        if savefig:
            if filename is None:
                filename = self.filename[:-4] + "_loop.png"
            self.fig.savefig(filename)

    def plot_derivative(self, df, filename=None):
        """Plot derivative of MOKE signal (susceptibility).
        
        The derivative shows the rate of magnetization change with field,
        useful for identifying magnetic transitions and features.
        
        Args:
            df (pd.DataFrame): Data frame containing 'Hx (Oe)' and 'Longitudinal Kerr (mV)' columns
            filename (str): Optional path to save the figure (default: None, displays only)
        """
        plt.figure(figsize=(5,4))
        # Calculate first derivative of signal
        dy = savgol_filter(self.data["Longitudinal Kerr (mV)"], 10, 2, deriv=1)
        # Calculate numerical differential of field
        dx = np.gradient(self.data["Hx (Oe)"]/10)
        # Compute derivative (dM/dH)
        derivative = dy/dx
        plt.plot(self.data["Hx (Oe)"]/10, derivative)
        
        plt.xlabel(r"$\mu_0 H$ (mT)")
        plt.ylabel(r"$d(M/M_s)/d(\mu_0 H)$")

        if filename is not None:
            plt.savefig(filename)