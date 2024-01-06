# It's a branch for industry
*The faiss version was added.//2024.01.06

*The changes are as follows:  

### 1.Use a ng instead of gt
~~~
class
├── test
│   ├── good
│   └── ng
├── train
    └── good  

~~~

### 2.Support Libtorch 
~~~
*Two files are stored after the test.
# extracted features , Instead of pickle
patchcore_features.pt 

# wide_resnet50_2
patchcore_model.pt 

*You can use c++ as follows.
vision::models::WideResNet50_2 module_wideresnet_50_;//https://github.com/pytorch/vision
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

 auto embedding_concat = [](torch::Tensor x, torch::Tensor y) {
         int64 B1 = x.size(0), C1 = x.size(1), H1 = x.size(2), W1 = x.size(3);
         int64 B2 = y.size(0), C2 = y.size(1), H2 = y.size(2), W2 = y.size(3);
         int64 s = H1 / H2;

         x = F::unfold(x, F::UnfoldFuncOptions(s).dilation(1).stride(s));
         x = x.view({B1, C1, -1, H2, W2});
         auto z = torch::zeros({B1, C1 + C2, x.size(2), H2, W2});
         for (int i = 0; i < x.size(2); i++) {
           auto temp = x.index({Slice(None, None), Slice(None, None), i,
                                Slice(None, None), Slice(None, None)});
           z.index({Slice(None, None), Slice(None, None), i, Slice(None, None),
                    Slice(None, None)}) = torch::cat({temp, y}, 1);
         }

         z = z.view({B1, -1, H2 * W2});
         z = F::fold(z, F::FoldFuncOptions({H1, W1}, {s, s}).stride(s))
                 .to(at::kCUDA);

         return z;
       };
 
auto embedding_vectors = embedding_concat(embed1, embed2);

//reshape_embedding
embedding_vectors.squeeze_();
embedding_vectors = embedding_vectors.reshape(
   {embedding_vectors.size(0),
    embedding_vectors.size(1) * embedding_vectors.size(2)});
embedding_vectors = embedding_vectors.permute({1, 0});


int p = 2;
int k = 9;
auto dist =  torch::cdist(embedding_vectors, anomaly_features, p);
auto knn = std::get<0>(dist.topk(k, -1, false));
int block_size =static_cast<int>(std::sqrt(knn.size(0)));
auto anomaly_map = knn.index({Slice(None, None), 0}).reshape({block_size, block_size});
double max_score = cfg_.vanomaly[category].anomalyMaxScore;
double min_score = cfg_.vanomaly[category].anomalyMinScore;
auto scores = (anomaly_map - min_score) / (max_score - min_score);

auto scores_resized =
   F::interpolate(
       scores.unsqueeze(0).unsqueeze(0),
       F::InterpolateFuncOptions()
           .size(std::vector<int64_t>{cfg_.origin_height, cfg_.origin_width})
           .align_corners(false)
           .mode(torch::kBilinear))
       .squeeze()
       .squeeze();

auto anomaly_mat = tensor2dToMat(scores_resized.to(at::kCPU));

cv::Mat anomaly_colormap, anomaly_mat_scaled;
anomaly_mat.at<float>(0, 0) = 1;
anomaly_mat.convertTo(anomaly_mat_scaled, CV_8UC3, 255.f);

applyColorMap(anomaly_mat_scaled, anomaly_colormap, cv::COLORMAP_JET);
cv::Mat anomaly_mat_origin_size;
cv::resize(anomaly_colormap, anomaly_mat_origin_size, {cfg_.origin_height, cfg_.origin_width});
auto origin_mat = get_origin_image_buffers()[batch_idx];

cv::resize(origin_mat, anomaly_mat_origin_size, {cfg_.origin_height, cfg_.origin_width});
cv::Mat dst;
cv::addWeighted(anomaly_mat_origin_size, 0.5, anomaly_colormap, 1 - 0.5, 0, dst);

...

~~~

### 3.Support Faiss 
~~~
If use faiss version of the main branch
Refer to PatchCore.cpp
~~~

### 4.Use all data for heatmap  
*We can see the difference between NG and GOOD.  
![image](https://user-images.githubusercontent.com/17777591/130405811-7d29432f-5be2-4c5b-a324-d95f526bb725.png)
![image](https://user-images.githubusercontent.com/17777591/130405756-371c582f-6c8c-4f46-bc6d-5e572b9a1ccc.png)
 



### 5.Using ROC Curve for best threshold detection
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

