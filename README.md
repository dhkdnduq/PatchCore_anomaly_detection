# It's a branch for industry
*The changes are as follows:  

### 1.Use a ng instead of gt
~~~
├── test
│   ├── good
│   └── ng
├── train
    └── good  

~~~

### 2.Support Libtorch 
~~~
*Two files are saved.
# extracted features , Instead of pickle
patchcore_features.pt 

# wide_resnet50_2
patchcore_model.pt 

*You can use c++ as follows.
*https://github.com/pytorch/vision

vision::models::WideResNet50_2 module_wideresnet_50_;
auto anomaly_features = torch::jit::load("patchcore_features.pt");
anomaly_features.attr("feature").toTensor().to(at::kCUDA);

torch::load(module_wideresnet_50_, "patchcore_model.pt");
module_wideresnet_50_->eval();
module_wideresnet_50_->to(at::kCUDA);

auto inputs = get_inputs();//image tensor
auto x = module_wideresnet_50_->conv1->forward(inputs);
x = module_wideresnet_50_->bn1->forward(x).relu_();
x = torch::max_pool2d(x, 3, 2, 1);

*instead of register_forward_hook
auto outputs1 = module_wideresnet_50_->layer1->forward(x);
auto outputs2 = module_wideresnet_50_->layer2->forward(outputs1);
auto outputs3 = module_wideresnet_50_->layer3->forward(outputs2);

auto m = AvgPool2d(AvgPool2dOptions(3).stride(1).padding(1));
auto embed1 = m(outputs2);
auto embed2 = m(outputs3);

auto embedding_vectors = embedding_concat(embed1, embed2);

Continue to implement using libtorch.
...

~~~


### 3.Use all data for heatmap  
*We can see the difference between NG and GOOD.  
![image](https://user-images.githubusercontent.com/17777591/130405811-7d29432f-5be2-4c5b-a324-d95f526bb725.png)
![image](https://user-images.githubusercontent.com/17777591/130405756-371c582f-6c8c-4f46-bc6d-5e572b9a1ccc.png)
 



### 4.Using ROC Curve for best threshold detection
*Find the threshold value with the mean value of the anomaly map.  
![image](https://user-images.githubusercontent.com/17777591/130405911-2c6077d0-80d8-41ba-914f-9683f0ac926f.png)


# PatchCore anomaly detection
Unofficial implementation of PatchCore(new SOTA) anomaly detection model


Original Paper : 
Towards Total Recall in Industrial Anomaly Detection (Jun 2021)  
Karsten Roth, Latha Pemula, Joaquin Zepeda, Bernhard Schölkopf, Thomas Brox, Peter Gehler  


https://arxiv.org/abs/2106.08265  
https://paperswithcode.com/sota/anomaly-detection-on-mvtec-ad

![plot](./capture/capture.jpg)


### Usage 
~~~
# install python 3.6, torch==1.8.1, torchvision==0.9.1
pip install -r requirements.txt

python train.py --phase train or test --dataset_path .../mvtec_anomaly_detection --category carpet --project_root_path path/to/save/results --coreset_sampling_ratio 0.01 --n_neighbors 9'

# for fast try just specify your dataset_path and run
python train.py --phase test --dataset_path .../mvtec_anomaly_detection --project_root_path ./
~~~

