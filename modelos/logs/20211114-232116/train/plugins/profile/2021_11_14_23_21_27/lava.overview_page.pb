? $	K?Hʌ??S?e????.??i?!??rf?B??	!       "Y
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
H???*	!       2	!       :	?d???y?H?|	?d??!kH?c?C??B	!       J	!       R	!       Z	!       b	!       JGPUb qq??ee?F@y?	a??K@?"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1_grad/MatMulMatMul,???N???!,???N???0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_1_grad/MatMulMatMul?H??RI??!?????n??0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_grad/MatMulMatMul ?:6??!??S??ĳ?0"|
`sequential_2/sequential/lstm/while/body/_1/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1MatMul?j????!?` ø?"?
lsequential_2/sequential_1/lstm_2/while/body/_327/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_1MatMul?	h?̹??!lc?:????"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/split_grad/concatConcatV2z??ב?!?M?????"?
?gradient_tape/sequential_2/sequential/lstm_1/while/sequential_2/sequential/lstm_1/while_grad/body/_670/gradient_tape/sequential_2/sequential/lstm_1/while/gradients/sequential_2/sequential/lstm_1/while/lstm_cell_1/split_grad/concatConcatV2$JOfɑ?!*7^??L??"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/split_grad/concatConcatV2{?\ Z???!9?i?؃??"?
?gradient_tape/sequential_2/sequential/lstm_1/while/sequential_2/sequential/lstm_1/while_grad/body/_670/gradient_tape/sequential_2/sequential/lstm_1/while/gradients/sequential_2/sequential/lstm_1/while/lstm_cell_1/MatMul_grad/MatMulMatMul<??ͨ???!??'?M???0"?
fsequential_2/sequential/lstm_1/while/body/_159/sequential_2/sequential/lstm_1/while/lstm_cell_1/MatMulMatMul߂/?o???!<|m?;???0Q      Y@YH??? ??a??7?X@qҸ????W@y?-?C|?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?45.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?95.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 