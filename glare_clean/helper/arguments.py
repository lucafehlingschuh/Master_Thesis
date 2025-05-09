def get_parameter(spec: dict, attr: str, default, dtype: type):
    if spec is not None and attr_in_spec(spec, attr):
        attr = get_actual_attr(spec, attr)
        parameter = spec[attr]
        if dtype == bool:
            if isinstance(parameter, str):
                parameter = parameter in ["true", "True", "1"]
            return dtype(parameter)
        else:
            if isinstance(parameter, dtype):
                return parameter
            else:
                return default

    # attr does not exist, return default
    return default


def attr_in_spec(spec, attr):
    attr = attr.lower()
    return any(k.lower() == attr for k in spec)


def get_actual_attr(spec, attr):
    lower_attr = attr.lower()
    for k in spec:
        if k.lower() == lower_attr:
            return k  
    return None  