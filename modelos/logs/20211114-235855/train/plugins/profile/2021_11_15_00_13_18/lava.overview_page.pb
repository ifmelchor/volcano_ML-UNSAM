?$	A?xI?J???V?նe???]i??t?!*?Z^?޺?	!       "Y
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
	!       	!       "$	?%`a?2??S?u?%?t?????8r?!?g?,{??*	!       2	!       :	?=Ab?{??g?è!??!+??O8???B	!       J	!       R	!       Z	!       b	!       JGPUb q"U?&?R@y?w?~d?8@?"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_1_grad/MatMulMatMul,\?:lޘ?!,\?:lޘ?0"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1_grad/MatMulMatMul.?W?ɘ?!??Ө?0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_grad/MatMulMatMul%???L??! A??ɱ?0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/split_grad/concatConcatV255M???!m?-????"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/split_grad/concatConcatV2K?$???!0T Jz???"?
?gradient_tape/sequential_2/sequential/lstm_1/while/sequential_2/sequential/lstm_1/while_grad/body/_670/gradient_tape/sequential_2/sequential/lstm_1/while/gradients/sequential_2/sequential/lstm_1/while/lstm_cell_1/split_grad/concatConcatV2????В?!??ח@'??"?
?gradient_tape/sequential_2/sequential/lstm_1/while/sequential_2/sequential/lstm_1/while_grad/body/_670/gradient_tape/sequential_2/sequential/lstm_1/while/gradients/sequential_2/sequential/lstm_1/while/lstm_cell_1/MatMul_grad/MatMulMatMul?C(\???!?e?X???0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_1_grad/MatMul_1MatMulےY?]???!???4|???"|
`sequential_2/sequential/lstm/while/body/_1/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1MatMul???????!???3????"?
fsequential_2/sequential/lstm_1/while/body/_159/sequential_2/sequential/lstm_1/while/lstm_cell_1/MatMulMatMule???#???!R?zs????0Q      Y@Y??"???a?Xw8?X@q??;3*?W@y?u?/v?}?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?75.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 