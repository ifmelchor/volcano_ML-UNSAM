	?EB[?e???EB[?e??!?EB[?e??	??h،2@??h،2@!??h،2@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?EB[?e???߆?y??1??^?2???I?v??-???Y#???S???r0*	???Q?R@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?
?lw??!y?H??9E@)????I??1?B?wօC@:Preprocessing2T
Iterator::Root::ParallelMapV2?]??Nw??!fC???3@)?]??Nw??1fC???3@:Preprocessing2E
Iterator::Root?$???!qw??A@)o??;????1؞i{0@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??X???!??s??%@)???{b}?1??z??"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip???I???!pGDy?P@)???~1{?1?????!@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap.u?׃I??!??͛?HF@)T?^PZ?1??Qh?? @:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor(?XQ?iX?!|fIx??)(?XQ?iX?1|fIx??:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?E&??HR?!??G????)?E&??HR?1??G????:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?y?ؘ?Q?!?[VG ??)?y?ؘ?Q?1?[VG ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 18.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.high"?64.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9??h،2@I????P@QiuX??.@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?߆?y???߆?y??!?߆?y??      ??!       "	??^?2?????^?2???!??^?2???*      ??!       2      ??!       :	?v??-????v??-???!?v??-???B      ??!       J	#???S???#???S???!#???S???R      ??!       Z	#???S???#???S???!#???S???b      ??!       JGPUY??h،2@b q????P@yiuX??.@