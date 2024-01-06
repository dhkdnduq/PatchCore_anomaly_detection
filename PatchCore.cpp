#include "pch.h"
#include "DLImpl.h"


#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/index_factory.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuClonerOptions.h>
#include <faiss/gpu/StandardGpuResources.h>

#include <vector>

using faiss::Index;
using faiss::gpu::GpuClonerOptions;
using faiss::gpu::GpuMultipleClonerOptions;
using faiss::gpu::GpuResourcesProvider;


struct PatchCoreParams {

    float anomalyMaxScore = 0.f;
    float anomalyMinScore = 0.f;
    float anomalyThreshold = 0.f;

};

template <typename T = vision::models::WideResNet50_2, typename I = at::Tensor>
class PatCoreDL : public DLImpl<T, I, PatCoreDL<T, I>>
{
public:
    PatCoreDL(DLModelParam* pParams) :DLImpl(pParams) {}
    bool load(string filepath)
    {

        filesystem::path realpath(filepath);
        realpath = realpath.replace_extension(".json");

        std::ifstream ifs;
        ifs.open(realpath.c_str(), ios_base::in);
        if (!ifs.is_open())
            return false;

        stringstream ss;
        string errors;
        ss << ifs.rdbuf();

        Json::CharReaderBuilder builder;
        std::unique_ptr<Json::CharReader> const reader(builder.newCharReader());
        string resnetModelPath;
        Json::Value root;
        bool bparsed = reader->parse(ss.str().c_str(), ss.str().c_str() + ss.str().length(), &root, &errors);

        if (!bparsed) return false;


        Json::Value& jcommon = root["common"];
        vecInputDims = { 1, 3, 224, 224 };
        if (jcommon) {
            this->input_width = std::stoi(jcommon.get("width", "100").asString().c_str());
            this->input_height = std::stoi(jcommon.get("height", "100").asString().c_str());
            this->input_batch_size = std::stoi(jcommon.get("batchsize", "1").asString().c_str());
            this->input_channel = std::stoi(jcommon.get("channel", "3").asString().c_str());
            this->nb = std::stoi(jcommon.get("nb", "784").asString().c_str());//default 224 * 224
            vecInputDims[0] = input_batch_size;
            vecInputDims[1] = input_channel;
            vecInputDims[2] = input_width;
            vecInputDims[3] = input_height;
            resnetModelPath = jcommon.get("modelpath", "").asString().c_str();
        }
        Json::Value& janomaly = root["anomaly"];

        if (janomaly) {
            faiss_pca_index.clear();
            for (auto feature : janomaly["features"]) {
                PatchCoreParams param;
                string featurePath = feature.get("featurepath", "").asString();
                try
                {
                    if (!filesystem::exists(featurePath))
                        return false;

                    auto p = new faiss::gpu::StandardGpuResources();
                    auto idx = faiss::read_index(featurePath.c_str());
                    auto gpu_res = reinterpret_cast<faiss::gpu::GpuResourcesProvider*>(p);
                    faiss::gpu::GpuClonerOptions* options = reinterpret_cast<faiss::gpu::GpuClonerOptions*>(new faiss::gpu::GpuClonerOptions());;
                    auto res = reinterpret_cast<faiss::gpu::GpuResourcesProvider*>(gpu_res);
                    faiss::Index* gpu_index = faiss::gpu::index_cpu_to_gpu(
                        res,
                        0,
                        idx,
                        reinterpret_cast<const faiss::gpu::GpuClonerOptions*>(options));

                    p->noTempMemory();
                    delete[] idx;

                    faiss_pca_index.emplace_back(unique_ptr<faiss::Index>(gpu_index));

                    param.anomalyMaxScore =
                        std::stof(feature.get("max_score", "32").asString().c_str());
                    param.anomalyMinScore =
                        std::stof(feature.get("min_score", "7").asString().c_str());
                    param.anomalyThreshold =
                        std::stof(feature.get("threshold", "20").asString().c_str());

                    feature_params.push_back(param);
                }
                catch (const c10::Error& e) {
                    std::cerr << " Error loading PatchCoreParams\n";
                    return false;
                }
                catch (faiss::FaissException& e)
                {
                    std::cerr << " FaissException" << e.what() << std::endl;;
                    return false;
                }
            }
        }
        ifs.close();

        try
        {
            if (torch::cuda::is_available()) {
                device_type = torch::kCUDA;
            }
            else {
                device_type = torch::kCPU;
            }
            torch::load(module_, resnetModelPath);
            to(device_type);
        }
        catch (const c10::Error& e) {
            std::cerr << " Error loading the PatchCore Model\n";
            return false;
        }
        catch (...)
        {
            return false;
        }
        return true;
    }
    void add_buffer(I input)
    {
        input_tensors.emplace_back(input);
    }
    void eval()
    {
        module_->eval();
    }
    void to(c10::DeviceType devType)
    {
        module_->to(devType);
    }
    at::Tensor PreProcessImage(cv::Mat frame, const cv::Scalar& resize, double scalefactor, const cv::Scalar& mean, const cv::Scalar& std)
    {
    	auto input_size = cv::Size(resize[0], resize[1]);
     
    	cv::Mat resized;
    	cv::resize(frame, resized, input_size, 0, 0);
    	cv::Mat flt_image;
    	if (frame.channels() == 1)
    		cv::cvtColor(resized, resized, cv::COLOR_GRAY2BGR);
    	resized.convertTo(flt_image, CV_32FC3, 1.f / scalefactor);
    	int channel = flt_image.channels();
    	// subtract mean
    	vector<cv::Mat> img_channels;
    	if (mean.cols != 0 && std.cols != 0) {
    		cv::Mat subtract, divide;
    		cv::subtract(flt_image,
    			cv::Scalar(mean[2], mean[1], mean[0]),
    			subtract);
    
    		cv::split(subtract, img_channels);
    		cv::divide(img_channels[0], std[2], img_channels[0]);
    		cv::divide(img_channels[1], std[1], img_channels[1]);
    		cv::divide(img_channels[2], std[0], img_channels[2]);
    
    		cv::merge(img_channels, flt_image);
    	}
    	int nChannels = 3;
    	Mat dmy(resize[1], resize[0], CV_32FC3);
    	torch::Tensor tensor_img;
    	tensor_img = torch::from_blob(dmy.data, { dmy.rows, dmy.cols, dmy.channels() }, torch::kFloat);
    	tensor_img = tensor_img.permute({ 2, 0, 1 });
    	tensor_img.unsqueeze_(0);
    	tensor_img = tensor_img.to(at::kCUDA);
    	return tensor_img;
    }

    
at::Tensor EmbeddingConcat(torch::Tensor x, torch::Tensor y) {

	int64 B1 = x.size(0), C1 = x.size(1), H1 = x.size(2), W1 = x.size(3);
	int64 B2 = y.size(0), C2 = y.size(1), H2 = y.size(2), W2 = y.size(3);
	int64 s = H1 / H2;

	x = F::unfold(x, F::UnfoldFuncOptions(s).dilation(1).stride(s));
	x = x.view({ B1, C1, -1, H2, W2 });
	auto z = torch::zeros({ B1, C1 + C2, x.size(2), H2, W2 });
	for (int i = 0; i < x.size(2); i++) {
		// z.index({Slice(None, None, i,None,None)}) = torch::cat(x.index({Slice(None, None, i),y}), 1);
		auto temp = x.index({ Slice(None, None), Slice(None, None), i, Slice(None, None),Slice(None, None) });
		z.index({ Slice(None, None), Slice(None, None), i, Slice(None, None),Slice(None, None) }) = torch::cat({ temp,y }, 1);
	}

	z = z.view({ B1, -1, H2 * W2 });
	z = F::fold(z, F::FoldFuncOptions({ H1, W1 }, { s, s }).stride(s)).to(at::kCUDA);

	return z;
}

    
   cv::Mat Convert2DTensorToMat(torch::Tensor x)
  {
    	int H = x.size(0);
    	int W = x.size(1);
    	int C = 1;
    
    	if (x.sizes().size() > 2)
    		C = x.size(2);
    
    	at::ScalarType type = x.scalar_type();
    	cv::Mat mat;
    	if (type == at::ScalarType::Float) {
    		mat = cv::Mat(H, W, C == 1 ? CV_32FC1 : CV_32FC3);
    		std::memcpy(mat.data, x.data_ptr(), sizeof(float) * x.numel());
    	}
    
    	else if (type == at::ScalarType::Char) {
    		mat = cv::Mat(H, W, C == 1 ? CV_8UC1 : CV_8UC3);
    		std::memcpy(mat.data, x.data_ptr(), sizeof(char) * x.numel());
    	}
    	else if (type == at::ScalarType::Bool) {
    		mat = cv::Mat(H, W, C == 1 ? CV_8UC1 : CV_8UC3);
    		std::memcpy(mat.data, x.data_ptr(), sizeof(char) * x.numel());
       }
       return mat;
   }
    torch::jit::IValue forward(std::vector<I> vecTensorInputs)
    {
        torch::jit::IValue rst;
        auto inputs = PreProcessImages(vecTensorInputs, cv::Scalar(224., 224.), 1., cv::Scalar(0.485, 0.456, 0.406), cv::Scalar(0.229, 0.224, 0.225));
        auto x = module_->conv1->forward(inputs);

        x = module_->bn1->forward(x).relu_();
        x = torch::max_pool2d(x, 3, 2, 1);

        auto outputs1 = module_->layer1->forward(x);
        auto outputs2 = module_->layer2->forward(outputs1);
        auto outputs3 = module_->layer3->forward(outputs2);
        auto m = AvgPool2d(AvgPool2dOptions(3).stride(1).padding(1));
        auto embed1 = m(outputs2);
        auto embed2 = m(outputs3);

        try {
            auto embedded = EmbeddingConcat(embed1, embed2);

            for (int batch_idx = 0; batch_idx < input_batch_size; batch_idx++)
            {
                embedded = embedded.squeeze(0);
                embedded = embedded.permute({ 1, 2, 0 }).contiguous();
                embedded = embedded.reshape({ embedded.size(0) * embedded.size(0), embedded.size(2) });
                int block_size = static_cast<int>(std::sqrt(embedded.size(0)));

                for (int feature_idx = 0; feature_idx < faiss_pca_index.size(); feature_idx++)
                {
                    std::unique_ptr<faiss::Index::idx_t[]> assign(new faiss::Index::idx_t[k_nn * nb]);
                    std::unique_ptr<float[]> dis(new float[k_nn * nb]);

                    auto embedded_tensor = torch::from_blob(dis.get(), { embedded.size(0),k_nn }, torch::kFloat);;
                    torch::cuda::synchronize();//Synchronization is required because the libraries are different.

                    faiss_pca_index[feature_idx]->search(nb, embedded.data_ptr<float>(), k_nn, dis.get(), assign.get());
                   
                    auto anomaly_map = embedded_tensor.index({ Slice(None, None), 0 }).reshape({ block_size,block_size });
                    
                    double max_score = feature_params[batch_idx].anomalyMaxScore;
                    double min_score = feature_params[batch_idx].anomalyMinScore;
                    auto scores = (anomaly_map - min_score) / (max_score - min_score);

                    auto scores_resized =
                        F::interpolate(scores.unsqueeze(0).unsqueeze(0),
                            F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{input_width,
                                input_height})
                            .align_corners(false)
                            .mode(torch::kBilinear)).squeeze().squeeze();

                    torch::cuda::synchronize();

                    auto anomaly_mat = Convert2DTensorToMat(scores_resized.to(at::kCPU));
                    cv::Mat anomaly_colormap, anomaly_mat_scaled;
                    anomaly_mat.at<float>(0, 0) = 1;
                    anomaly_mat.convertTo(anomaly_mat_scaled, CV_8UC3, 255.f);

                    applyColorMap(anomaly_mat_scaled, anomaly_colormap, cv::COLORMAP_JET);
                    
                    //only one batch size has been supported.
                    rst = scores_resized.to(at::kCPU);
                    
                }

            }
        }
        catch (const c10::Error& e) {
            std::cout << e.msg() << endl;
        }
        return rst;
    }

protected:
    vector<unique_ptr<faiss::Index>> faiss_pca_index;
    int k_nn{ 9 };
    int nb{ 784 };  // nb of queries
    vector< PatchCoreParams> feature_params;
};
