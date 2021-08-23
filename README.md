# It's a branch for personal backup
*The changes are as follows:  

### 1.Don't use GT 
~~~
├── test
│   ├── good
│   └── ng
├── train
│   ├── good  

~~~

### 2.Support Libtorch 
~~~
*Two files are saved.
# extracted features , Instead of pickle
patchcore_features.pt 

# wide_resnet50_2
patchcore_model.pt 

*You can use c++ as follows.
auto anomaly_features = torch::jit::load("patchcore_features.pt");
anomaly_features.attr("feature").toTensor().to(at::kCUDA);
 
torch::load(module_wideresnet_50_, "patchcore_model.pt");
module_wideresnet_50_->eval();
module_wideresnet_50_->to(at::kCUDA);
~~~


### 3.Use all data for heatmap  
*We can see the difference between NG and GOOD.

### 4.Using ROC Curve for best threshold detection
*Find the threshold value with the mean value of the anomaly map.

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

