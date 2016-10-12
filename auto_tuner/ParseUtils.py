def try_to_parse_int(value, defaultValue=0):
    try:
        return int(value)
    except ValueError:
        return defaultValue

def try_to_parse_bool(value):
    value = value.lower()
    if value == "false" or value == "0":
        return False
    else:
        return True

def try_to_parse_int_list(value, delimiter=","):
    elems = value.split(delimiter)
    list = []
    for elem in elems:
        try:
            list.append(int(elem))
        except ValueError:
            a = 0
    return list

def try_to_parse_str_list(value, delimiter=","):
    elems = value.split(delimiter)
    list = []
    for elem in elems:
        list.append(elem)
    return list
    