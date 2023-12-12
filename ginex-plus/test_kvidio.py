import cupy
import kvikio
import kvikio.defaults
import time 


MB = 2 ** 20
GB = 2 ** 30

kvikio.defaults.compat_mode_reset(False)
kvikio.defaults.reset_num_threads(8)
print (kvikio.defaults.compat_mode())

print (kvikio.defaults.get_num_threads())
a = cupy.arange(1e9)
f = kvikio.CuFile("/data01/liuyibo/test-file", "w")
# Write whole array to file
write_start = time.time()
f.write(a)
write_floating = time.time() - write_start
print ("time cost: {:.4f} s, transfer bandwidth: {:4f} GB/s ".format(write_floating, a.shape[0] * 8 / (write_floating * GB)))
f.close()

b = cupy.empty_like(a)
f = kvikio.CuFile("/data01/liuyibo/test-file", "r")
print (f.open_flags())
# Read whole array from file
read_start = time.time()
f.read(b)
read_floating = time.time() - read_start
print ("transfer bandwidth: {:4f} GB/s ".format(a.shape[0] * 8 / (read_floating * GB)))
# assert all(a == b)
# Use contexmanager
# c = cupy.empty_like(a)
# with kvikio.CuFile("/data01/liuyibo/test-file", "r") as f:
#     f.read(c)
# assert all(a == c)

# # Non-blocking read
# d = cupy.empty_like(a)
# with kvikio.CuFile("/data01/liuyibo/test-file", "r") as f:
#     future1 = f.pread(d[:50])
#     future2 = f.pread(d[50:], file_offset=d[:50].nbytes)
#     future1.get()  # Wait for first read
#     future2.get()  # Wait for second read
# assert all(a == d)

print ("=" * 20 + "finish" + "=" * 20)