	ywd?6???ywd?6???!ywd?6???		???A#@	???A#@!	???A#@"z
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails9ywd?6????????5??1??x??A??????`?Iw?$???Y?˷>?7??r0*	e;?O??U@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?B?5v???!?X?h?D@)???QI???1HU?EɟB@:Preprocessing2T
Iterator::Root::ParallelMapV2T??Yh???!?K???n7@)T??Yh???1?K???n7@:Preprocessing2E
Iterator::Root?dU?????!??:!?D@)%??1??1???RN'2@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?E?xx??!??b???$@)!?????1т????"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip%?s}??!D]???4M@)?????jo?1Nη???@:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensorj>"?Db?!@?Hf\z@)j>"?Db?1@?Hf\z@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap-$`ty??!???t??E@)?M?G??]?13+??`? @:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?HP?X?!??????)?HP?X?1??????:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorRH2?w?M?!%???i???)RH2?w?M?1%???i???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?67.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*moderate2t11.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9	???A#@I)CW|?S@Q?Z@SB(@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????5???????5??!?????5??      ??!       "	??x????x??!??x??*      ??!       2	??????`???????`?!??????`?:	w?$???w?$???!w?$???B      ??!       J	?˷>?7???˷>?7??!?˷>?7??R      ??!       Z	?˷>?7???˷>?7??!?˷>?7??b      ??!       JGPUY	???A#@b q)CW|?S@y?Z@SB(@