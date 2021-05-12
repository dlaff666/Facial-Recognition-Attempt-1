import dlib
dlib.cuda.set_device(0)
print(dlib.DLIB_USE_CUDA)
print(dlib.cuda.get_num_devices())