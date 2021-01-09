def read_data(file_name):
    with open(file_name, 'r') as f_o:
        data = f_o.readlines()
    return data
