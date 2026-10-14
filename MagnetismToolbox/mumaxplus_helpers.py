from __future__ import annotations
import numpy as np
from mumaxplus import Ferromagnet, Grid
from typing import Callable, Literal, Mapping, Sequence, TypeAlias
import numpy.typing as npt
from tqdm import tqdm


# Define some useful types
LoopDirection = Literal["up", "down", "up_down", "down_up"]
RelaxMethod = Literal["minimize", "relax", "none"]
WindowName = Literal["hann", "kaiser"]
QuantityValue: TypeAlias = (
    float | int | npt.ArrayLike
)
QuantityMap: TypeAlias = dict[
    str, Callable[[], QuantityValue]
]

def _field_path(hmin: float, hmax: float, hstep: float, sweep_direction: LoopDirection) -> np.ndarray:
    """Helper to generate an array of fields
    hmin (float): minimum field (T), e.g. -100e-3 (-100 mT)
    hmax (float): maximum field (T), e.g. 100e-3 (+100 mT)
    hstep (float): field step (T). Absolute value will be taken
    sweep_direction (str): "up", "down", "up_down", "down_up". 
                    "up" sweeps from hmin to hmax. "down" sweeps from hmax to hmin". 
                    "up_down"/"down_up" combines both sweeps
    """
    hstep = np.abs(hstep)
    if hmax < hmin:
        raise ValueError("Require hmax >= hmin.")

    # Function to create field path segment
    def segment(start: float, stop: float) -> np.ndarray:
        return np.linspace(start, stop, int(np.ceil(abs(stop - start) / hstep)) + 1)

    # Create field path
    if sweep_direction == "up":
        return segment(hmin, hmax)
    if sweep_direction == "down":
        return segment(hmax, hmin)
    if sweep_direction == "up_down":
        return np.concatenate((segment(hmin, hmax), segment(hmax, hmin)[1:]))
    if sweep_direction == "down_up":
        return np.concatenate((segment(hmax, hmin), segment(hmin, hmax)[1:]))
    raise ValueError("sweep_direction must be 'up', 'down', 'up_down', or 'down_up'.")

def _equilibrate(magnet: Ferromagnet, method: RelaxMethod, tolerance: float) -> None:
    """Helper method to equilibriate the sample using relax() or minimize()"""
    if method == "minimize":
        magnet.minimize(tol=tolerance)
    elif method == "relax":
        magnet.relax(tol=tolerance)
    elif method != "none":
        raise ValueError("equilibration must be 'minimize', 'relax', or 'none'.")

def create_ferromagnet(world, grid_shape, msat, aex, alpha,
                  name: str = "magnet",
                  initial_magnetization = (1.0, 1.0, 1.0),
                  anisotropy: Mapping[str, object] | None = None,
                  geometry = None,
                  regions = None,
                  ):
    """
    world: mumax World for the simulation
    grid_shape: iterable (nx, ny, nz), where nx, ny, nz are integers
    msat: saturation magnetisation in A/m. Scalar, ndarray (1, nz, ny, nx), or function that returns iterable of shape (1,) for given input x, y, z
    aex: exchange strength in J/m. Scalar, ndarray, or function
    alpha: damping (unitless). Scalar, ndarray, or function
    name: (str) magnet name
    initial_magnetization: iterable of shape (3,), or ndarray (3, nz, ny, nx) or function that returns iterable of shape (3,) for given input x, y, z
    anisotropy: Dictionary of (key) anisotopy property name and (value) anisotropy property value. Refer to mumax+ documentation.
    geometry: Magnet geometry. None, ndarray (bool) (1, nz, ny, nx) or function that returns bool for given input x, y, z
    regions: Defined regions of magnet. None, ndarray (int) (1, nz, ny, nx) or function that returns int for given input x, y, z
    """
    # Check grid_shape is correct form
    shape = tuple(int(n) for n in grid_shape)
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError("grid_shape must be three positive integers: (nx, ny, nz).")

    # Make the magnet with given calues
    magnet = Ferromagnet(world, Grid(shape), name=name, geometry=geometry, regions=regions)
    magnet.msat = msat
    magnet.aex = aex
    magnet.alpha = alpha
    magnet.magnetization = initial_magnetization

    # Set anisotropy values
    allowed = {"ku1", "ku2", "anisU", "kc1", "kc2", "kc3", "anisC1", "anisC2"}
    for key, value in (anisotropy or {}).items():
        if key not in allowed:
            raise ValueError(f"Unsupported anisotropy key {key!r}; allowed keys: {sorted(allowed)}")
        setattr(magnet, key, value)

    return magnet

def run_hysteresis(magnet, hmin, hmax, hstep,
                   virgin: bool = False,
                   sweep_direction: LoopDirection = "up_down",
                   field_direction = (1.0, 0.0, 0.0),
                   equilibriation_method: RelaxMethod = "relax",
                   tolerance: float = 1e-6,
                   output_quantities: QuantityMap | None = None,
                   use_tqdm: bool = False,
                   ):
    """ Runs a hysteresis loop with DC fields.
    magnet: magnet to run the hysteresis on.
    hmin, hmax, hstep (float): minimum, maximum and field steps. Requires hmin < hmax. Absolute value of hstep taken
    virgin (bool): Include virgin sweep from zero field
    direction (str): "up", "down", "up_down", "down_up". 
                        "up" sweeps from hmin to hmax. "down" sweeps from hmax to hmin". 
                        "up_down"/"down_up" combines both sweeps
    field_direction ((3,) iterable): direction of applied dc field.
    equilibriation_method: Method to relax/minimize the magnet at each step
    output_quantities: Dictionary of values to output. Values should be callable.
    use_tqdm (bool): Use tqdm progress bar
    """
    # Check sensible inputs
    if len(field_direction) != 3 :
            raise ValueError("field_direction must have shape 3: (hx, hy, hz).")
    
    # Setup fields for the loop
    fields = _field_path(hmin, hmax, hstep, sweep_direction)
    if virgin:
        virgin_path = np.linspace(0.0, fields[0], int(np.ceil(abs(fields[0]/hstep))) + 1)
        fields = np.concatenate((virgin_path, fields[1:]))

    # Setup arrays to hold output_quantities
    outputs = {}
    
    # Run through fields
    field_iterator = enumerate(tqdm(fields)) if use_tqdm else enumerate(fields)
    for i, field_magnitude in field_iterator:
        # Apply field
        magnet.bias_magnetic_field = tuple(field_magnitude * component for component in field_direction)
        # Equilibriate
        _equilibrate(magnet, equilibriation_method, tolerance)

        # Store output values
        for key, getter in output_quantities.items():
            # Execute the callable quantity at the current state. Store it as an ndarray
            value = np.asarray(getter())

            # Allocate storage from the first result
            if key not in outputs:
                outputs[key] = np.empty((len(fields), *value.shape),
                                        dtype=value.dtype)
            
            # Store the value
            outputs[key][i] = value

    return outputs

def run_fmr_broadband(magnet, fres, fmax, 
                      oversampling: float = 1.0,
                      drive_field: Callable = None,
                      drive_field_mask = None,
                      output_quantities: QuantityMap | None = None,
                      output_scripts = None,
                      use_tqdm: bool = False,
                    ):
    """Runs fmr for a given drive_field. Calculates time step and total time from fres, fmax and 
    oversampling. Only adds the drive_field to the existing field. Does not modify or overwrite any
    previously set field.
    magnet: mumax+ magnet to run fmr on
    fres: target frequency resolution in Hz
    fmax: maximum frequency in Hz
    oversampling (float): oversampling factor (default = 1, i.e. No oversampling)
    drive_field: User-defined drive_field. Should be callable with time t (in s)
    output_quantities: Dictionary of values to output. Values should be callable.
    use_tqdm (bool): Use tqdm progress bar
    """
    # Bring me the world
    world = magnet.world

    # Calculate timings
    dt = 1.0 / (2.0 * oversampling * fmax)
    t_tot = 1.0 / fres
    n_tsteps = int(np.ceil(t_tot/dt))
    tsteps = np.arange(0.0, t_tot + dt, dt)


    # Setup arrays to hold output_quantities, and put in t=0 values
    outputs = {}
    for key, getter in (output_quantities or {}).items():
        # Execute the callable quantity at the current state.
        value = np.asarray(getter())
        
        # Allocate storage from the initial state
        outputs[key] = np.empty((n_tsteps+1, *value.shape),
            dtype=value.dtype)

        # Store the initial value
        outputs[key][0] = value
    
    for key, script in (output_scripts or {}).items():
        script()

    # Apply the drive_field and run through time steps
    magnet.bias_magnetic_field.add_time_term(drive_field, mask=drive_field_mask)
    # time_iterator =  if use_tqdm else range(n_tsteps)
    with tqdm(range(n_tsteps), disable=not use_tqdm) as time_iterator:
        for i in time_iterator:
            # Run for the first time step
            world.timesolver.run(dt)

            for key, getter in (output_quantities or {}).items():
                # Execute the callable quantity at the current state. Store it as an ndarray
                value = np.asarray(getter())
                outputs[key][i+1] = value

            for key, script in (output_scripts or {}).items():
                script()
            

    # Remove the drive field
    magnet.bias_magnetic_field.remove_time_terms()

    return outputs
