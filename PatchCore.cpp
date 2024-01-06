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

    torch::jit::IValue forward(std::vector<I> vecTensorInputs)
    {
        torch::jit::IValue rst;
        auto inputs = TorchHelper::PreProcessImages(vecTensorInputs, cv::Scalar(224., 224.), 1., cv::Scalar(0.485, 0.456, 0.406), cv::Scalar(0.229, 0.224, 0.225));
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
            auto embedded = TorchHelper::EmbeddingConcat(embed1, embed2);

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

                    auto anomaly_mat = TorchHelper::Convert2DTensorToMat(scores_resized.to(at::kCPU));
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
