#property file

configs={}
configs['cifar10dir']='../datasets/cifar10'
configs['cifar10batchesdir']=configs['cifar10dir']+'/cifar-10-batches-py'
configs['tinycifar10hog.trainfile']=configs['cifar10dir']+'/tinycifar10-hog/features_tinycifar10_train.h5'
configs['tinycifar10hog.valfile']=configs['cifar10dir']+'/tinycifar10-hog/features_tinycifar10_val.h5'
configs['tinycifar10hog.testfile']=configs['cifar10dir']+'/tinycifar10-hog/features_tinycifar10_test.h5'
configs["devicename"]='/gpu:1'
