import pathlib


data_root = pathlib.Path('dataroot')
all_data = pathlib.Path('dataroot/alldata')
train_data = pathlib.Path('dataroot/traindata')
test_data = pathlib.Path('dataroot/testdata')
validation_data =pathlib.Path('dataroot/validation_data')


IMAGE_SIZE = 124

TrainPer = 0.7
TestPer =0.15
ValidationPer =0.15


# if not train_data.exists():
#     print('make dir train')
#     train_data.mkdir()
#     for dr in all_data.glob('*/'):
#         path =train_data.joinpath(dr.name)
#         pathlib.Path(path).mkdir()
#
# if not test_data.exists():
#     print('make dir test')
#     test_data.mkdir()
#     for dr in all_data.glob('*/'):
#         path =test_data.joinpath(dr.name)
#         pathlib.Path(path).mkdir()
#
# if not validation_data.exists():
#     print('make dir validation')
#     validation_data.mkdir()
#     for dr in all_data.glob('*/'):
#         path =validation_data.joinpath(dr.name)
#         pathlib.Path(path).mkdir()