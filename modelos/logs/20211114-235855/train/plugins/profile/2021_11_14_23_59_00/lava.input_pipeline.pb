$	!a売???lR?Qړ?*6?u?!{?!GW??:??	!       "Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails;*6?u?!{?1*6?u?!{?r59"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!<,?V]?j??1$Di?}?I??ut\??r60"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails=?A`??"{?1?A`??"{?r61"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails>d???^D{?1d???^D{?r62"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!???ĭ???16w??\?V?I???????r63"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!@GW??:??1??6?4D??I@??wԘ??r64*	?z?G?Y@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?.????!?Ɂ???J@)m<?b?Ϫ?1b?z#?RI@:Preprocessing2T
Iterator::Root::ParallelMapV2??_ ???!?=??|?0@)??_ ???1?=??|?0@:Preprocessing2E
Iterator::Root??<????!????t>@)uv28J^??1??B ?+@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat????Ç?!%??i*r&@)?{?i????1B6??"@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip?s?Lh??!?{???bQ@))	????q?1???yJ@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??IӠh^?!?䃎???)??IӠh^?1?䃎???:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor
??O?mY?!(/?4B??)
??O?mY?1(/?4B??:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??T?????!?????K@)?&OYM?S?1|2^?b???:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceiUMuO?!7??????)iUMuO?17??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?62.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI^?y??rO@Q?'?kD?B@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	`??U?׀???K2ȶz?6w??\?V?!??6?4D??*	!       2	!       :	?ut\????'?1???!@??wԘ??B	!       J	!       R	!       Z	!       b	!       JGPUb q^?y??rO@y?'?kD?B@