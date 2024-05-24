import os
import sys

# try:
import manage_gpus as gpl
have_gpu_manager = True
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# except (ImportError, ModuleNotFoundError):
#     have_gpu_manager = False

print('have gpu manager : ', have_gpu_manager)
def lock_gpu(cpu):
    if not cpu:
        if have_gpu_manager:
            try:
                gpuid = gpl.get_gpu_lock(gpu_device_id=-1, soft=False)
                print("GPU locked", gpuid, file=sys.stderr)
                return "GPU", gpuid
            except gpl.NoGpuManager:
                import warnings
                warnings.warn("no gpu manager available")
                del os.environ["CUDA_VISIBLE_DEVICES"]
            except gpl.NoGpuAvailable:
                print("no GPU available - aborting program", file=sys.stderr)
                sys.exit(1)
        else:
            return "GPU (if available)"
    else:
        print("No GPU requested, force using CPU", file=sys.stderr)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "CPU"
