$	K?Hʌ??S?e????.??i?!??rf?B??	!       "Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails8??.??i?1??.??i?r56"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!9t|?8c???1]?E?~U?I??Udt@??r57"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??H?}}?1??H?}}?r58"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails;1zn?+??11zn?+??r59"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails<1zn?+??11zn?+??r60"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails=C?8
??1C?8
??r61"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails>1{?v???11{?v???r62"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails?C?8
??1C?8
??r63"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails@U?????1U?????r64"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!A??rf?B??1??Z
H???IkH?c?C??r65*	?G?z?R@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?-t%՟?!?8[?e?D@)l???f??1]H??C@:Preprocessing2T
Iterator::Root::ParallelMapV2?up?71??!????yD:@)?up?71??1????yD:@:Preprocessing2E
Iterator::Root<??.???!D??E@)?n?HJ??1?8?1@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?r??+|?!{?C/RR"@)J'L5?v?1t?&??@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??>eĥ?!??6??PL@)i?????k?1WD?5Y@:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor?^?sa?W?!I?T[???)?^?sa?W?1I?T[???:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensors.?Ue?U?!?C??t??)s.?Ue?U?1?C??t??:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap?@.q䁠?!??IUyE@)?3??k?R?1???J????:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicetϺFˁN?!???????)tϺFˁN?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?45.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIq??ee?F@Q?	a??K@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	?1}
۪?z?Y"i?n?]?E?~U?!??Z
H???*	!       2	!       :	?d???y?H?|	?d??!kH?c?C??B	!       J	!       R	!       Z	!       b	!       JGPUb qq??ee?F@y?	a??K@