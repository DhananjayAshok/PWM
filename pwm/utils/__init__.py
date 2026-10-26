# Utilities shared across the project. Import from here: `from pwm.utils import load_parameters, log_info`.
# Simulator environments (setup/envs/*) import this package too and carry only pyyaml/numpy, so the plotting and
# statistics helpers (seaborn, pandas) are imported on first use.
from pwm.utils.parameter_handling import load_parameters
from pwm.utils.log_handling import log_error, log_info, log_warn, log_dict
from pwm.utils.hash_handling import write_meta, add_meta_details
from pwm.utils.fundamental import file_makedir


def __getattr__(name):
    if name == "Plotter":
        from pwm.utils.plot_handling import Plotter
        return Plotter
    if name == "paired_bootstrap":
        from pwm.utils.tests import paired_bootstrap
        return paired_bootstrap
    raise AttributeError(f"module 'pwm.utils' has no attribute {name!r}")
