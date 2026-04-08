from os.path import join

class FLAGES(object):

    lr = 4e-4
    decay_rate = 0.99
    decay_step = 10000
    num_workers = 4
    train_batch_size = 4
    test_batch_size = 1
    val_batch_size = 1

    trainA_path = r""  #raw
    trainB_path = r""  #reference

    valA_path = r""  #raw
    valB_path = r""  #reference
    testA_path = r""  #raw

    out_path = r"/results/output_test/"  

    model_dir = r'/results/model_save/'

    backup_model_dir = join(model_dir)
