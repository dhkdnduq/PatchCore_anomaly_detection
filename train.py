import argparse
import torch
from torch.nn import functional as F
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import glob
import shutil
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
import gc
import csv
import time
from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from torchvision.models import wide_resnet50_2


class TrainFeature(torch.jit.ScriptModule):
    #__constants__ = ['feature']

    def __init__(self,feature_):
        super(TrainFeature, self).__init__()
        #self.feature = feature_
        self.register_buffer('feature', feature_)
    def forward(self):
        pass


def copy_files(src, dst, ignores=[]):
    src_files = os.listdir(src)
    for file_name in src_files:
        ignore_check = [True for i in ignores if i in file_name]
        if ignore_check:
            continue
        full_file_name = os.path.join(src, file_name)
        if os.path.isfile(full_file_name):
            shutil.copy(full_file_name, os.path.join(dst,file_name))
        if os.path.isdir(full_file_name):
            os.makedirs(os.path.join(dst, file_name), exist_ok=True)
            copy_files(full_file_name, os.path.join(dst, file_name), ignores)

def prep_dirs(root):
    # make embeddings dir
    # embeddings_path = os.path.join(root, 'embeddings')
    embeddings_path = os.path.join('./', 'embeddings', args.category)
    os.makedirs(embeddings_path, exist_ok=True)
    # make sample dir
    sample_path = os.path.join(root, 'sample')
    os.makedirs(sample_path, exist_ok=True)
    # make source code record dir & copy
    source_code_save_path = os.path.join(root, 'src')
    os.makedirs(source_code_save_path, exist_ok=True)
    #copy_files('./', source_code_save_path, ['.git','.vscode','__pycache__','logs','README','samples','LICENSE']) # copy source code
    return embeddings_path, sample_path, source_code_save_path

def embedding_concat(x, y):
    # from https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z

def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])

    return embedding_list

#imagenet
mean_train = [0.485, 0.456, 0.406]
std_train = [0.229, 0.224, 0.225]

class MVTecDataset(Dataset):
    def __init__(self, root, transform,  phase):
        if phase=='train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')

        self.transform = transform
        # load dataset
        self.img_paths,  self.labels, self.types = self.load_dataset() # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(self.img_path)
        img_types = ('*.png','*.bmp','*.jpg')

        for defect_type in defect_types:
            img_paths = []
            if defect_type == 'good':
                for img_type in img_types:
                    img_paths.extend(glob.glob(os.path.join(self.img_path, defect_type) + '/'+ img_type))

                img_tot_paths.extend(img_paths)
                tot_labels.extend([0]*len(img_paths))
                tot_types.extend(['good']*len(img_paths))
            else:
                for img_type in img_types:
                    img_paths.extend(glob.glob(os.path.join(self.img_path, defect_type) + '/' + img_type))

                img_paths.sort()
                img_tot_paths.extend(img_paths)
                tot_labels.extend([1]*len(img_paths))
                tot_types.extend([defect_type]*len(img_paths))


        return img_tot_paths, tot_labels, tot_types

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img,label, os.path.basename(img_path[:-4]), img_type


def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def heatmap_on_image(heatmap, image):
    if heatmap.shape != image.shape:
        heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
    out = np.float32(heatmap)/255 + np.float32(image)/255
    out = out / np.max(out)
    return np.uint8(255 * out)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)    


def cal_confusion_matrix(y_true, y_pred_no_thresh, thresh, img_path_list):
    pred_thresh = []
    false_n = []
    false_p = []
    for i in range(len(y_pred_no_thresh)):
        if y_pred_no_thresh[i] > thresh:
            pred_thresh.append(1)
            if y_true[i] == 0:
                false_p.append(img_path_list[i])
        else:
            pred_thresh.append(0)
            if y_true[i] == 1:
                false_n.append(img_path_list[i])

    cm = confusion_matrix(y_true, pred_thresh)
    print(cm)
    print('false positive')
    print(false_p)
    print('false negative')
    print(false_n)

def randomize_tensor(tensor):
    return tensor[torch.randperm(len(tensor))]


def distance_matrix(x, y=None, p=2):  # pairwise distance of vectors

    y = x if type(y) == type(None) else y

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, p).sum(2)

    return dist


class NN():

    def __init__(self, X=None, Y=None, p=2):
        self.p = p
        self.train(X, Y)

    def train(self, X, Y):
        self.train_pts = X
        self.train_label = Y

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        if type(self.train_pts) == type(None) or type(self.train_label) == type(None):
            name = self.__class__.__name__
            raise RuntimeError(f"{name} wasn't trained. Need to execute {name}.train() first")

        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)
        labels = torch.argmin(dist, dim=1)
        return self.train_label[labels]
class KNN(NN):

    def __init__(self, X=None, Y=None, k=3, p=2):
        self.k = k
        super().__init__(X, Y, p)

    def train(self, X, Y):
        super().train(X, Y)
        if type(Y) != type(None):
            self.unique_labels = self.train_label.unique()

    def predict(self, x):

        test = 1 / self.p
        dist = distance_matrix(x, self.train_pts, self.p) ** (1 / self.p)

        knn = dist.topk(self.k, largest=False)


        return knn


class STPM(pl.LightningModule):
    def __init__(self, hparams):
        super(STPM, self).__init__()

        self.save_hyperparameters(hparams)

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)
        #self.model = wide_resnet50_2(pretrained=True, progress=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()

        self.data_transforms = transforms.Compose([
                        transforms.Resize((args.input_size, args.input_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])


        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.img_path_list = []
        self.mean_score_norm = []
        self.all_scores = []
        self.all_scores_mean_norm = []
        self.image_batch_list = []
        self.x_type_list =[]
        self.y_true = []
    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def save_anomaly_map(self, anomaly_map, input_img, gt_img, file_name, x_type):
        if anomaly_map.shape != input_img.shape:
            anomaly_map = cv2.resize(anomaly_map, (input_img.shape[0], input_img.shape[1]))
        anomaly_map_norm = min_max_norm(anomaly_map)
        anomaly_map_norm_hm = cvt2heatmap(anomaly_map_norm*255)

        # anomaly map on image
        heatmap = cvt2heatmap(anomaly_map_norm*255)
        hm_on_img = heatmap_on_image(heatmap, input_img)

        # save images
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}.jpg'), input_img)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap.jpg'), anomaly_map_norm_hm)
        cv2.imwrite(os.path.join(self.sample_path, f'{x_type}_{file_name}_amap_on_img.jpg'), hm_on_img)

    def train_dataloader(self):
        image_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=args.batch_size, shuffle=True, num_workers=0) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=self.data_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=0) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_list = []
    
    def on_test_start(self):
        self.init_results_list()
        self.embedding_dir_path, self.sample_path, self.source_code_save_path = prep_dirs(self.logger.log_dir)
        self.embedding_coreset = pickle.load(open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'rb'))
        embeded = torch.tensor(self.embedding_coreset)
        train_jit = TrainFeature(embeded)
        traced_model = torch.jit.script(train_jit)
        torch.jit.save(traced_model, "patchcore_features.pt")
        dummy_input = torch.randn(args.batch_size, 3, args.input_size, args.input_size, requires_grad=False).cuda()
        traced_script_module = torch.jit.trace(self.model, dummy_input)
        traced_script_module.save("patchcore_model.pt")


        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, file_name, _ = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding = embedding_concat(embeddings[0], embeddings[1])
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))
        gc.collect()

    def training_epoch_end(self, outputs):
        total_embeddings = np.array(self.embedding_list)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*args.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        print('initial embedding size : ', total_embeddings.shape)
        print('final embedding size : ', self.embedding_coreset.shape)
        with open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'wb') as f:
            pickle.dump(self.embedding_coreset, f)
        gc.collect()

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search

        x, label, file_name, x_type = batch
        features = self(x)
        embeddings = []
        for feature in features:
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature))
        embedding_ = embedding_concat(embeddings[0], embeddings[1])
        embedding_test = np.array(reshape_embedding(np.array(embedding_)))
   
        # NN
        knn = KNN(torch.from_numpy(self.embedding_coreset).cuda(),k=9)
        score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()
        self.img_path_list.extend(file_name)
        # support multi input size 
        block_size = int(np.sqrt(len(score_patches)))
        anomaly_map = score_patches[:, 0].reshape((block_size, block_size))
        self.all_scores.append(anomaly_map)
        self.image_batch_list.append(x)
        self.x_type_list.append(x_type)
        self.y_true.append(label.cpu().numpy()[0])



    def Find_Optimal_Cutoff(self,target,predicted):
        fpr,tpr,threshold = roc_curve(target,predicted,pos_label=1)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        return list(roc_t['threshold']), threshold
        '''
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], '--', color='black')  
        plt.title('ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
        '''

    def analyze_data(self):
        score_pathces = np.array(self.all_scores)
        for i,val in enumerate(score_pathces):
            self.all_scores_mean_norm.append(np.mean(val))

        min_score = np.min(score_pathces)
        max_score = np.max(score_pathces)
        scores = (score_pathces - min_score) / (max_score - min_score)
        for i,heatmap in  enumerate(scores):
            anomaly_map_resized = cv2.resize(heatmap, (args.input_size, args.input_size))
            max_ = np.max(heatmap)
            min_ = np.min(heatmap)

            anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
            anomaly_map_resized_blur[0][0] = 1.



            # save images
            x = self.image_batch_list[i]
            x = self.inv_normalize(x)
            input_x = cv2.cvtColor(x.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            if anomaly_map_resized_blur.shape != input_x.shape:
                anomaly_map_resized_blur = cv2.resize(anomaly_map_resized_blur, (input_x.shape[0], input_x.shape[1]))

            # anomaly map on image
            heatmap = cvt2heatmap(anomaly_map_resized_blur * 255)
            hm_on_img = heatmap_on_image(heatmap, input_x)

            # save images
            cv2.imwrite(os.path.join(self.sample_path, f'{self.x_type_list[i]}_{self.img_path_list[i]}.jpg'), input_x)
            cv2.imwrite(os.path.join(self.sample_path, f'{self.x_type_list[i]}_{self.img_path_list[i]}_amap.jpg'), heatmap)
            cv2.imwrite(os.path.join(self.sample_path, f'{self.x_type_list[i]}_{self.img_path_list[i]}_amap_on_img.jpg'), hm_on_img)


    def test_epoch_end(self, outputs):
        self.analyze_data()

        best_th, threshold = self.Find_Optimal_Cutoff(self.y_true, self.all_scores_mean_norm)
        print(f'\nbest threshold={best_th}')
        ng_index = np.where(np.array(self.y_true) == 1)
        if len(ng_index[0]) == 0:
            ng_index = len(self.y_true)
        else:
            ng_index = ng_index[0][0]
        fig = plt.figure()
        sns.histplot(self.all_scores_mean_norm[:ng_index], kde=True, color="blue", label="normal")
        sns.histplot(self.all_scores_mean_norm[ng_index:], kde=True, color="red", label="abnormal")
        fig.legend(labels=['normal', 'abnormal'])
        plt.xlabel("Anomaly score")
        plt.ylabel("Count")
        plt.savefig('Anomaly_score_histplot.jpg')


def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='train')
    parser.add_argument('--dataset_path', default=r'/home/changwoo/hdd/datasets/mvtec_anomaly_detection2') # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--category', default='bottle')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--load_size', default=256) # 256
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.001)
    parser.add_argument('--project_root_path', default=r'D:/deeplearning/PatchCore_anomaly_detection/') # 'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=9)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()

    trainer = pl.Trainer.from_argparse_args(args, default_root_dir=os.path.join(args.project_root_path, args.category), max_epochs=args.num_epochs, gpus=1) #, check_val_every_n_epoch=args.val_freq,  num_sanity_val_steps=0) # ,fast_dev_run=True)
    model = STPM(hparams=args)
    if args.phase == 'train':
        trainer.fit(model)
        #trainer.test(model)
    elif args.phase == 'test':
        trainer.test(model)

