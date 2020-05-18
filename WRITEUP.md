# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

Any layer which is not natively supported by a given model framework is classified as **custom layer** by model optimiser. 

Custom layers are a necessary and important to have feature of the OpenVINO™ Toolkit, although we shouldn’t have to use it very often, if at all, due to all of the supported layers. 

The [list of supported layers](https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) from earlier very directly relates to whether a given layer is a custom layer. Any layer not in that list is automatically classified as a custom layer by the Model Optimizer.

To actually add custom layers, there are a few differences depending on the original model framework. In both TensorFlow and Caffe, the first option is to register the custom layers as extensions to the Model Optimizer.

For Caffe, the second option is to register the layers as Custom, then use Caffe to calculate the output shape of the layer. You’ll need Caffe on your system to do this option.

For TensorFlow, its second option is to actually replace the unsupported subgraph with a different subgraph. The final TensorFlow option is to actually offload the computation of the subgraph back to TensorFlow during inference.



## Comparing Model Performance

### Open SSD Inception V2 Coco Model

I downloaded and extracted the SSD Mobile Net model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md):

```bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
mkdir -p models; cd models
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
tar xvf ssd_incep*
```

Convert the model to IR:

```bash
cd ssd_inc*
export OVDT="/opt/intel/openvino/deployment_tools"
$OVDT/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config $OVDT/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

Conversion took 63 seconds.

To run:

```bash
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_inception_v2_coco_2018_01_28//frozen_inference_graph.xml -l $OVDT/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

Inference time for this model was around 165 ms.

### Open SSD Mobilenet V2 Coco Model

I downloaded and extracted the SSD Mobile Net model from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md):

```bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
mkdir -p models; cd models
wget 
http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar xvf ssd_mob*
```

Then we convert fronzen graph to IR:

```bash
cd ssd_mob*
export OVDT="/opt/intel/openvino/deployment_tools"
$OVDT/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config $OVDT/model_optimizer/extensions/front/tf/ssd_v2_support.json
```

Conversion takes around 76 seconds.

To run the app:

```bash
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l $OVDT/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

The inference time of the model post-conversion was 67 ms.

### Faster RCNN Inception V2 Coco Model

I downlad the model from from [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)::

```bash
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
mkdir -p models; cd models
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar xvf faster_*
```

Then we convert fronzen graph to IR:

```bash
cd faster_*
export OVDT="/opt/intel/openvino/deployment_tools"
python $OVDT/model_optimizer/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_use_custom_operations_config  $OVDT/model_optimizer/extensions/front/tf/faster_rcnn_support.json --data_type   FP32 --reverse_input_channels --input_shape "[1, 300, 300, 3]" --tensorflow_object_detection_api_pipeline_config pipeline.config
```

Conversio took 140 seconds. To run:

```bash
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.xml -d CPU -pt 0.5 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

```

Getting the error:

```
Invalid buffer size, packet size 130 < expected frame_size 995328
```



### Person Detection Retail 0013 in Open Vino

I then used the person detector model from [here](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html).

To download the model:

```bash
python  $OVDT/tools/model_downloader/downloader.py --name person-detection-retail-0013 -o /home/workspace/models/pre_trained/
```

To run the app:

```bash
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/pre_trained/intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

The inference time of this model was ~ 46 ms.

To sum up, I see open vino model has much better inference time, but Open SSD Coco model also worked fine.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are: 

- checking the number of people in the office. This can be used to help in case of emergency issues in the building. 
- In case of security situation, say bank locker, we can track how many users are inside and since how long.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. 

### Effect of Lighting

- Lighting may affect the edge detection by the models as it may not be able to find the user or may wrongfully classify one as the person.

### Model Accuracy

The selection of the model is the most important step in this whole exercise. If we get it wrong, no matter how accurate the model is, it will not bear any good results. In any case, if we choose a good model, model accuracy helps us in detecting the people better. I see that the coco model was performing nice, but sometimes had issue with the black color. Open Vino person detector model was doing good.

### Focal length and image size

The image quality will affect the real time processing of the image. High resolution camera will generate good quality image, but than it depends on the compute unit in the edge device. 