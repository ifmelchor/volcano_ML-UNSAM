	a??????a??????!a??????	?ꍠ/?#@?ꍠ/?#@!?ꍠ/?#@"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9a???????N?z1???13???U???AP?c*??I??m?????Y??6o???r0*	?A`??]@2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?=?>tA??!:??3i?8@)?=?>tA??1:??3i?8@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate@1?d????!?	??I@)??c??3??1a???~8@:Preprocessing2E
Iterator::Root??k	????!P????C@)??鲘؜?1?h?s28@:Preprocessing2T
Iterator::Root::ParallelMapV2a??_Yi??!??p?.@)a??_Yi??1??p?.@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?i???z?!?!.??@)HP?sw?1ƀթ??@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip1?0&???!?P?-N@)?1??ln?1?OM??	@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap1?闈???!?Ce?3?I@)#??fF?Z?1q>G?[??:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???V_]U?!?z??????)???V_]U?1?z??????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?iT?dK?!?b"????)?iT?dK?1?b"????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?72.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9?ꍠ/?#@IJ?*, ?R@Q?J??-@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?N?z1????N?z1???!?N?z1???      ??!       "	3???U???3???U???!3???U???*      ??!       2	P?c*??P?c*??!P?c*??:	??m???????m?????!??m?????B      ??!       J	??6o?????6o???!??6o???R      ??!       Z	??6o?????6o???!??6o???b      ??!       JGPUY?ꍠ/?#@b qJ?*, ?R@y?J??-@