?$	?i4???$??1X????Z?QfS?!^??-???	!       "Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:M?n?{?1M?n?{?r58"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails;?'eRC{?1?'eRC{?r59"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails<M?n?{?1M?n?{?r60"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails=?'eRC{?1?'eRC{?r61"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails>?'eRC{?1?'eRC{?r62"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails??k$	?u?1?k$	?u?r63"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails@*6?u?!{?1*6?u?!{?r64"b
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails!A^??-???1V??W9r?I??[X7޽?r65"Y
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails_?Z?QfS?1?Z?QfS?r95*	p=
ף@R@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate$|?o?^??!??N?{?C@)????B??1Kmw?7?A@:Preprocessing2T
Iterator::Root::ParallelMapV2&r?????!]q?7Cg9@)&r?????1]q?7Cg9@:Preprocessing2E
Iterator::Root????Q??!7?$b?D@)Wya???1鵌?|0@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?=~o??!?r???H%@)?????w?1/?????@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip)?Ǻ???!?R۝?M@)??1ZGUs?1??O?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??z`?! ZK??@)??z`?1 ZK??@:Preprocessing2?
MIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[1]::FromTensor9% &?B^?!?? ?=@)9% &?B^?1?? ?=@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???G???!e??d(?D@)??f?v?T?13????u??:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice>[{S?!^?0???)>[{S?1^?0???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.high"?69.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI??l??`Q@Q	?MB?|>@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	!       	!       "$	??;ߞIw??z??^??Z?QfS?!M?n?{?*	!       2	!       :	\:5????????:z???!??[X7޽?B	!       J	!       R	!       Z	!       b	!       JGPUb q??l??`Q@y	?MB?|>@?"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1_grad/MatMulMatMul?ZR??,??!?ZR??,??0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_1_grad/MatMulMatMulG;?6?ݘ?!?k?B??0"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_grad/MatMulMatMul???(@??!sTq???0"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/split_grad/concatConcatV2'#?S????!ѧa??̶?"?
?gradient_tape/sequential_2/sequential_1/lstm_2/while/sequential_2/sequential_1/lstm_2/while_grad/body/_495/gradient_tape/sequential_2/sequential_1/lstm_2/while/gradients/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/split_grad/concatConcatV2??'_???!??+?ݳ??"?
?gradient_tape/sequential_2/sequential/lstm_1/while/sequential_2/sequential/lstm_1/while_grad/body/_670/gradient_tape/sequential_2/sequential/lstm_1/while/gradients/sequential_2/sequential/lstm_1/while/lstm_cell_1/split_grad/concatConcatV2eB??Tݒ?!.~?5??"?
?gradient_tape/sequential_2/sequential/lstm/while/sequential_2/sequential/lstm/while_grad/body/_845/gradient_tape/sequential_2/sequential/lstm/while/gradients/sequential_2/sequential/lstm/while/lstm_cell/MatMul_1_grad/MatMul_1MatMul?z,Cu??!B??????"?
?gradient_tape/sequential_2/sequential/lstm_1/while/sequential_2/sequential/lstm_1/while_grad/body/_670/gradient_tape/sequential_2/sequential/lstm_1/while/gradients/sequential_2/sequential/lstm_1/while/lstm_cell_1/MatMul_grad/MatMulMatMulI ????!G???N???0"?
fsequential_2/sequential/lstm_1/while/body/_159/sequential_2/sequential/lstm_1/while/lstm_cell_1/MatMulMatMul?­?D???!v?J<????0"?
lsequential_2/sequential_1/lstm_2/while/body/_327/sequential_2/sequential_1/lstm_2/while/lstm_cell_2/MatMul_1MatMul=gh_O???!?9A2????Q      Y@Y??"???a?Xw8?X@q?ўG?.X@y?x???}?"?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
high?69.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*?Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb?96.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"Nvidia GPU (Turing)(: B 