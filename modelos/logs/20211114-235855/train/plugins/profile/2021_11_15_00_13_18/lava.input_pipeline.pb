$	A?xI?J???V?նe???]i??t?!*?Z^?޺?	!       "Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails<pD??k?|?1pD??k?|?r60"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails=?h㈵?t?1?h㈵?t?r61"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!>?=?N????1D??<???I?l?????r62"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??]i??t?1?]i??t?r63"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails@?g?,{??1?g?,{??r64"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!A*?Z^?޺?1????8r?I+??O8???r65*	 ?rh?5R@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??(#. ??!n ??pC@)?;???1?o=X??A@:Preprocessing2T
Iterator::Root::ParallelMapV2???p<???!U>?Aq?7@)???p<???1U>?Aq?7@:Preprocessing2E
Iterator::Root?W?L????!?,l???D@)
?Ƿw??1s??w1@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat5??,??!???.](@)2r????10??KK%@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip<??????!ӓqAtM@)F?n?1p?1?D???@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapL5??Ҟ?!????D?D@)|??8G]?1?h?X~?@:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorv?A]?PV?!ejZg:???)v?A]?PV?1ejZg:???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorܠ?[;QR?!E?V!???)ܠ?[;QR?1E?V!???:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?$?@Q?!>??ԏ!??)?$?@Q?1>??ԏ!??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?75.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI"U?&?R@Q?w?~d?8@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	?%`a?2??S?u?%?t?????8r?!?g?,{??*	!       2	!       :	?=Ab?{??g?è!??!+??O8???B	!       J	!       R	!       Z	!       b	!       JGPUb q"U?&?R@y?w?~d?8@