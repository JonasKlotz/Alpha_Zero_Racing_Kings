import tensorflow as tf
import keras
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

@timing
def time_test(worker):
	worker.train(epochs=10)

if __name__ == "__main__":
	print("Executing Test Run")
	print("===== TF Session =====")
	#tf.Session()

	print("===== Checking GPUS =====")
	tf.config.list_physical_devices('GPU')
	#tf.test.is_gpu_available() # True/False
	#tf.test.is_gpu_available(cuda_only=True) 
	print(tf.python.client.device_lib.list_local_devices())
	#keras.backend.tensorflow_backend._get_available_gpus()

	print("===== Running Script =====")
	from Model.worker import Worker
	worker = Worker('Model/config.yaml')
	worker.load_dummy_data()
	
	print("===== Execution time =====")
	print("10 Epochs time:")
	time_test(worker)
	#print(timeit.timeit("time_test()", setup="from __main__ import time_test"))
