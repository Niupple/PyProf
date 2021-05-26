

def is_sgemm(name: str):
    return 'sgemm' in name

def is_hgemm(name: str):
    return 'hgemm' in name

def is_data_movement(name: str):
    raise NotImplementedError