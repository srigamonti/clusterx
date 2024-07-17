import toml
import inspect

def read_toml_config(toml_file):
    """Read options from a TOML file."""
    with open(toml_file, 'r') as file:
        config = toml.load(file)
    return config

def get_command_name():
    return inspect.stack()[1].function

def inspect_function_signature(func, verbose=False):
    signature = inspect.signature(func)

    # Initialize lists for parameters with and without default values
    params_without_defaults = []
    params_with_defaults = []
    defaults = []

    # Iterate over the parameters
    for name, param in signature.parameters.items():
        if param.default == inspect.Parameter.empty:
            params_without_defaults.append(name)
        else:
            params_with_defaults.append(name)
            defaults.append(param.default)

    if verbose:
        # Print the results
        print("Parameters without default values:", params_without_defaults)
        print("Parameters with default values:", params_with_defaults)
        print("Default values:", defaults)

    return params_without_defaults, params_with_defaults, defaults


def dict_to_argv(params_without_defaults, params_with_defaults, defaults, param_dict):
    """ Transforms dict to argv list for use in plac
    params_without_defaults = ['p1', 'p2']
    params_with_defaults = ['p3', 'p4']
    defaults = ['v3', 'v4']
    param_dict = {'p1': 'v1', 'p3': 'v3', 'p2': 'v2'}

    output = generate_output(params_without_defaults, params_with_defaults, defaults, param_dict)
    print(output)  # Output: ['v1', 'v2', '--p3', 'v3', '--p4', 'v4']
    """
    output = []

    # Add parameters without defaults
    for param in params_without_defaults:
        if param in param_dict:
            output.append(param_dict[param])

    # Add parameters with defaults
    for param, default in zip(params_with_defaults, defaults):
        if param in param_dict:
            val = str(param_dict[param])
            print(default, val)
            if str(default) == "False" or str(default) == "True":
                if val == "False":
                    pass 
                else:
                    output.append(f'--{param.replace("_", "-")}')
            else:
                output.append(f'--{param.replace("_", "-")}')
                output.append(val)
    
    return output

def generate_argv(func, config_dict):
    params_without_defaults, params_with_defaults, defaults = inspect_function_signature(func)
    argv_list = dict_to_argv(params_without_defaults, params_with_defaults, defaults, config_dict)
    return argv_list

