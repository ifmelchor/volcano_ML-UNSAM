?$	!a売???lR?Qړ?*6?u?!{?!GW??:??	!       "Y
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
	!       	!       "$	`??U?׀???K2ȶz?6w??\?V?!??6?4D??*	!       2	!       :	?ut\????'?1???!@??wԘ??B	!       J	!       R	!       Z	!       b	!       JGPUb q^?y??rO@y?'?kD?B@?"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1_grad/MatMulMatMul?|?A????!?|?A????0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_1_grad/MatMulMatMul?!8?????!N??l????0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_grad/MatMulMatMulf+?ɰ??!?r?z???0"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/split_grad/concatConcatV2?,?????!???????"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/split_grad/concatConcatV2L???Г?!?; ?????"?
?gradient_tape/sequential_2/sequential/lstm_1/while/sequential_2/sequential/lstm_1/while_grad/body/_670/gradient_tape/sequential_2/sequential/lstm_1/while/gradients/sequential_2/sequential/lstm_1/while/lstm_cell_1/split_grad/concatConcatV20???m??!????Q??"|
`sequential_2/sequential/lstm/while/body/_1/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1MatMul??6?ic??!??? H??"?
?gradient_tape/sequential_2/sequential/lstm_1/while/sequential_2/sequential/lstm_1/while_grad/body/_670/gradient_tape/sequential_2/sequential/lstm_1/while/gradients/sequential_2/sequential/lstm_1/while/lstm_cell_1/MatMul_grad/MatMulMatMul?A?D?H??!?K?,??0"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1_grad/MatMul_1MatMul?VxB?	??!j?(?H??"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_1_grad/MatMul_1MatMul?o?i[??!j??g????Q      Y@Y??"???a?Xw8?X@q[Ƀ4_T@y?q?}???"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?62.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?81.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 