
orig_maes = {'ETTh1': 0.379, 'ETTh2':0.288,'ETTm1': 0.229}

orig_param = {'ETTh1': 
                {'epochs' : 10,
                 'batch_size' : 8,
                 'hid_size' : 4,
                 'num_levels' : 3,
                 'kernel_size' : 5,
                 'dropout' : 0.5,
                 'lr_rate' : 0.003,},
             'ETTh2':
                 {'epochs' : 10,
                 'batch_size' : 16,
                 'hid_size' : 8,
                 'num_levels' : 3,
                 'kernel_size' : 5,
                 'dropout' : 0.25,
                 'lr_rate' : 0.007,},
             'ETTm1' :
                 {'epochs' : 10,
                 'batch_size' : 32,
                 'hid_size' : 4,
                 'num_levels' : 3,
                 'kernel_size' : 5,
                 'dropout' : 0.5,
                 'lr_rate' : 0.005,}}