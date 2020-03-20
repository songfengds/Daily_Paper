# Daily_Paper
This project is for reading a paper everyday

## 2020.01 readed
### [diffGrad: An Optimization Method for Convolutional Neural Networks](https://arxiv.org/abs/1909.11015)
*Shiv Ram Dubey, Soumendu Chakraborty, Swalpa Kumar Roy, Snehasis Mukherjee, Satish Kumar Singh, Bidyut Baran Chaudhuri*
> (Submitted on 12 Sep 2019 (v1), last revised 6 Mar 2020 (this version, v3))

Stochastic Gradient Decent (SGD) is one of the core techniques behind the success of deep neural networks. The gradient provides information on the direction in which a function has the steepest rate of change. The main problem with basic SGD is to change by equal sized steps for all parameters, irrespective of gradient behavior. Hence, an efficient way of deep network optimization is to make adaptive step sizes for each parameter. Recently, several attempts have been made to improve gradient descent methods such as AdaGrad, AdaDelta, RMSProp and Adam. These methods rely on the square roots of exponential moving averages of squared past gradients. Thus, these methods do not take advantage of local change in gradients. In this paper, a novel optimizer is proposed based on the difference between the present and the immediate past gradient (i.e., diffGrad). In the proposed diffGrad optimization technique, the step size is adjusted for each parameter in such a way that it should have a larger step size for faster gradient changing parameters and a lower step size for lower gradient changing parameters. The convergence analysis is done using the regret bound approach of online learning framework. Rigorous analysis is made in this paper over three synthetic complex non-convex functions. The image categorization experiments are also conducted over the CIFAR10 and CIFAR100 datasets to observe the performance of diffGrad with respect to the state-of-the-art optimizers such as SGDM, AdaGrad, AdaDelta, RMSProp, AMSGrad, and Adam. The residual unit (ResNet) based Convolutional Neural Networks (CNN) architecture is used in the experiments. The experiments show that diffGrad outperforms other optimizers. Also, we show that diffGrad performs uniformly well for training CNN using different activation functions. The source code is made publicly available at this https [URL](https://github.com/shivram1987/diffGrad).

文章提出SGD的问题在于迭代过程中相等的步长，不会根据梯度表现自适应。AdaGrad、AdaDelta、RMSProp、Adam都在提高梯度下降的方法，然而这些方法依赖于梯度平方的指数移动平均的平方跟，并没有利用到梯度的局部变化。本文提出了一种新的diffGrad，梯度下降的步长会自适应变化，并给出了收敛行证明。并公开了代码。

diffGrad将当前和过去迭代的梯度差异（即短期梯度变化信息）与Adam优化技术结合在一起来控制优化过程的学习率。diffGrad在梯度变化大的时候得到一个更高的学习率，在梯度变化小的地方得到一个低的学习率。为了避免陷入局部最优或者鞍点，由惯性冲量moment来控制。

作者是几个IEEE，文中有其收敛性证明，先码起来，周末再研究一下，他人的学习笔记可以先参考：https://youyou-tech.com/2019/12/28/%E8%AE%A4%E8%AF%86DiffGrad%EF%BC%9A%E6%96%B0%E5%9E%8B%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E4%BC%98%E5%8C%96%E5%99%A8/


### [Image Segmentation Using Deep Learning: A Survey](https://arxiv.org/abs/2001.05566)
*Shervin Minaee, Yuri Boykov, Fatih Porikli, Antonio Plaza, Nasser Kehtarnavaz, Demetri Terzopoulos*
> (Submitted on 15 Jan 2020 (v1), last revised 18 Jan 2020 (this version, v2))

Image segmentation is a key topic in image processing and computer vision with applications such as scene understanding, medical image analysis, robotic perception, video surveillance, augmented reality, and image compression, among many others. Various algorithms for image segmentation have been developed in the literature. Recently, due to the success of deep learning models in a wide range of vision applications, there has been a substantial amount of works aimed at developing image segmentation approaches using deep learning models. In this survey, we provide a comprehensive review of the literature at the time of this writing, covering a broad spectrum of pioneering works for semantic and instance-level segmentation, including fully convolutional pixel-labeling networks, encoder-decoder architectures, multi-scale and pyramid based approaches, recurrent networks, visual attention models, and generative models in adversarial settings. We investigate the similarity, strengths and challenges of these deep learning models, examine the most widely used datasets, report performances, and discuss promising future research directions in this area.


他人的解读： [纽约大学发布「深度学习图像分割」最新综述论文，带你全面了解100个10大类深度图像分割算法](https://mp.weixin.qq.com/s?__biz=MzU2OTA0NzE2NA==&mid=2247519342&idx=1&sn=ace274849f033eee582276a505181daf&chksm=fc866d7dcbf1e46bc78c5d801bac8f9798929375459ffbd824002f633e9d8b992c984912a845&mpshare=1&scene=1&srcid=&sharer_sharetime=1579425668727&sharer_shareid=c37ff2288ffba696afb85e962505b352&exportkey=A9xRjmvz7Wko1JSHfKjG2kw%3D&pass_ticket=EI%2FDFK6tPBtIaDETwZirN5nq7eNSd%2Fo3sGmtuu1W2cJyeMbALbTZrpzcGHaiit13#rd)

----

## Paper Set

### [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415)Swish
*Dan Hendrycks, Kevin Gimpel*
> (Submitted on 27 Jun 2016 (v1), last revised 11 Nov 2018 (this version, v3))

We propose the Gaussian Error Linear Unit (GELU), a high-performing neural network activation function. The GELU nonlinearity is the expected transformation of a stochastic regularizer which randomly applies the identity or zero map to a neuron's input. The GELU nonlinearity weights inputs by their magnitude, rather than gates inputs by their sign as in ReLUs. We perform an empirical evaluation of the GELU nonlinearity against the ReLU and ELU activations and find performance improvements across all considered computer vision, natural language processing, and speech tasks.


### [Learning to segment images with classification labels](https://arxiv.org/abs/1912.12533)
*Ozan Ciga, Anne L. Martel*
> (Submitted on 28 Dec 2019)

Two of the most common tasks in medical imaging are classification and segmentation. Either task requires labeled data annotated by experts, which is scarce and expensive to collect. Annotating data for segmentation is generally considered to be more laborious as the annotator has to draw around the boundaries of regions of interest, as opposed to assigning image patches a class label. Furthermore, in tasks such as breast cancer histopathology, any realistic clinical application often includes working with whole slide images, whereas most publicly available training data are in the form of image patches, which are given a class label. We propose an architecture that can alleviate the requirements for segmentation-level ground truth by making use of image-level labels to reduce the amount of time spent on data curation. In addition, this architecture can help unlock the potential of previously acquired image-level datasets on segmentation tasks by annotating a small number of regions of interest. In our experiments, we show using only one segmentation-level annotation per class, we can achieve performance comparable to a fully annotated dataset.


### [Graph-FCN for image semantic segmentation](https://arxiv.org/abs/2001.00335)
*Yi Lu, Yaran Chen, Dongbin Zhao, Jianxin Chen*
> (Submitted on 2 Jan 2020)

Semantic segmentation with deep learning has achieved great progress in classifying the pixels in the image. However, the local location information is usually ignored in the high-level feature extraction by the deep learning, which is important for image semantic segmentation. To avoid this problem, we propose a graph model initialized by a fully convolutional network (FCN) named Graph-FCN for image semantic segmentation. Firstly, the image grid data is extended to graph structure data by a convolutional network, which transforms the semantic segmentation problem into a graph node classification problem. Then we apply graph convolutional network to solve this graph node classification problem. As far as we know, it is the first time that we apply the graph convolutional network in image semantic segmentation. Our method achieves competitive performance in mean intersection over union (mIOU) on the VOC dataset(about 1.34% improvement), compared to the original FCN model.


### [Deeper Insights into Weight Sharing in Neural Architecture Search](https://arxiv.org/abs/2001.01431)
OpenReviewer: https://openreview.net/forum?id=ryxmrpNtvH
*Yuge Zhang, Zejun Lin, Junyang Jiang, Quanlu Zhang, Yujing Wang, Hui Xue, Chen Zhang, Yaming Yang*
> (Submitted on 6 Jan 2020)

With the success of deep neural networks, Neural Architecture Search (NAS) as a way of automatic model design has attracted wide attention. As training every child model from scratch is very time-consuming, recent works leverage weight-sharing to speed up the model evaluation procedure. These approaches greatly reduce computation by maintaining a single copy of weights on the super-net and share the weights among every child model. However, weight-sharing has no theoretical guarantee and its impact has not been well studied before. In this paper, we conduct comprehensive experiments to reveal the impact of weight-sharing: (1) The best-performing models from different runs or even from consecutive epochs within the same run have significant variance; (2) Even with high variance, we can extract valuable information from training the super-net with shared weights; (3) The interference between child models is a main factor that induces high variance; (4) Properly reducing the degree of weight sharing could effectively reduce variance and improve performance.


### [Bridging the gap between AI and Healthcare sides: towards developing clinically relevant AI-powered diagnosis systems](https://arxiv.org/abs/2001.03923)
*Changhee Han, Leonardo Rundo, Kohei Murao, Takafumi Nemoto, Hideki Nakayama, Shin'ichi Satoh*
> (Submitted on 12 Jan 2020)

This work aims to identify/bridge the gap between Artificial Intelligence (AI) and Healthcare sides in Japan towards developing medical AI fitting into a clinical environment in five years. Moreover, we attempt to confirm the clinical relevance for diagnosis of our research-proven pathology-aware Generative Adversarial Network (GAN)-based medical image augmentation: a data wrangling and information conversion technique to address data paucity. We hold a clinically valuable AI-envisioning workshop among 2 Medical Imaging experts, 2 physicians, and 3 Healthcare/Informatics generalists. A qualitative/quantitative questionnaire survey for 3 project-related physicians and 6 project non-related radiologists evaluates the GAN projects in terms of Data Augmentation (DA) and physician training. The workshop reveals the intrinsic gap between AI/Healthcare sides and its preliminary solutions on Why (i.e., clinical significance/interpretation) and How (i.e., data acquisition, commercial deployment, and safety/feeling safe). The survey confirms our pathology-aware GANs' clinical relevance as a clinical decision support system and non-expert physician training tool. Radiologists generally have high expectations for AI-based diagnosis as a reliable second opinion and abnormal candidate detection, instead of replacing them. Our findings would play a key role in connecting inter-disciplinary research and clinical applications, not limited to the Japanese medical context and pathology-aware GANs. We find that better DA and expert physician training would require atypical image generation via further GAN-based extrapolation.


### [NAS evaluation is frustratingly hard](https://arxiv.org/abs/1912.12522)
*Antoine Yang, Pedro M. Esperança, Fabio M. Carlucci*
> (Submitted on 28 Dec 2019 (v1), last revised 13 Feb 2020 (this version, v3))

Neural Architecture Search (NAS) is an exciting new field which promises to be as much as a game-changer as Convolutional Neural Networks were in 2012. Despite many great works leading to substantial improvements on a variety of tasks, comparison between different methods is still very much an open issue. While most algorithms are tested on the same datasets, there is no shared experimental protocol followed by all. As such, and due to the under-use of ablation studies, there is a lack of clarity regarding why certain methods are more effective than others. Our first contribution is a benchmark of 8 NAS methods on 5 datasets. To overcome the hurdle of comparing methods with different search spaces, we propose using a method's relative improvement over the randomly sampled average architecture, which effectively removes advantages arising from expertly engineered search spaces or training protocols. Surprisingly, we find that many NAS techniques struggle to significantly beat the average architecture baseline. We perform further experiments with the commonly used DARTS search space in order to understand the contribution of each component in the NAS pipeline. These experiments highlight that: (i) the use of tricks in the evaluation protocol has a predominant impact on the reported performance of architectures; (ii) the cell-based search space has a very narrow accuracy range, such that the seed has a considerable impact on architecture rankings; (iii) the hand-designed macro-structure (cells) is more important than the searched micro-structure (operations); and (iv) the depth-gap is a real phenomenon, evidenced by the change in rankings between 8 and 20 cell architectures. To conclude, we suggest best practices, that we hope will prove useful for the community and help mitigate current NAS pitfalls. The code used is available at this https [URL](https://github.com/antoyang/NAS-Benchmark).


### [PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193)
*Alexander Kirillov, Yuxin Wu, Kaiming He, Ross Girshick*
> (Submitted on 17 Dec 2019 (v1), last revised 16 Feb 2020 (this version, v2))

We present a new method for efficient high-quality image segmentation of objects and scenes. By analogizing classical computer graphics methods for efficient rendering with over- and undersampling challenges faced in pixel labeling tasks, we develop a unique perspective of image segmentation as a rendering problem. From this vantage, we present the PointRend (Point-based Rendering) neural network module: a module that performs point-based segmentation predictions at adaptively selected locations based on an iterative subdivision algorithm. PointRend can be flexibly applied to both instance and semantic segmentation tasks by building on top of existing state-of-the-art models. While many concrete implementations of the general idea are possible, we show that a simple design already achieves excellent results. Qualitatively, PointRend outputs crisp object boundaries in regions that are over-smoothed by previous methods. Quantitatively, PointRend yields significant gains on COCO and Cityscapes, for both instance and semantic segmentation. PointRend's efficiency enables output resolutions that are otherwise impractical in terms of memory or computation compared to existing approaches. Code has been made available at this https [URL](https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend).


### [Non-local U-Net for Biomedical Image Segmentation](https://arxiv.org/abs/1812.04103)
*Zhengyang Wang, Na Zou, Dinggang Shen, Shuiwang Ji*
> (Submitted on 10 Dec 2018 (v1), last revised 18 Feb 2020 (this version, v2))

Deep learning has shown its great promise in various biomedical image segmentation tasks. Existing models are typically based on U-Net and rely on an encoder-decoder architecture with stacked local operators to aggregate long-range information gradually. However, only using the local operators limits the efficiency and effectiveness. In this work, we propose the non-local U-Nets, which are equipped with flexible global aggregation blocks, for biomedical image segmentation. These blocks can be inserted into U-Net as size-preserving processes, as well as down-sampling and up-sampling layers. We perform thorough experiments on the 3D multimodality isointense infant brain MR image segmentation task to evaluate the non-local U-Nets. Results show that our proposed models achieve top performances with fewer parameters and faster computation.


### [Deep Snake for Real-Time Instance Segmentation](https://128.84.21.199/abs/2001.01629?context=cs)
*Sida Peng, Wen Jiang, Huaijin Pi, Xiuli Li, Hujun Bao, Xiaowei Zhou*
> (Submitted on 6 Jan 2020 (v1), last revised 27 Feb 2020 (this version, v2))

This paper introduces a novel contour-based approach named deep snake for real-time instance segmentation. Unlike some recent methods that directly regress the coordinates of the object boundary points from an image, deep snake uses a neural network to iteratively deform an initial contour to the object boundary, which implements the classic idea of snake algorithms with a learning-based approach. For structured feature learning on the contour, we propose to use circular convolution in deep snake, which better exploits the cycle-graph structure of a contour compared against generic graph convolution. Based on deep snake, we develop a two-stage pipeline for instance segmentation: initial contour proposal and contour deformation, which can handle errors in initial object localization. Experiments show that the proposed approach achieves state-of-the-art performances on the Cityscapes, Kins and Sbd datasets while being efficient for real-time instance segmentation, 32.3 fps for 512×512 images on a 1080Ti GPU. The code will be available at this https [URL](https://github.com/zju3dv/snake/).


### [FarSee-Net: Real-Time Semantic Segmentation by Efficient Multi-scale Context Aggregation and Feature Space Super-resolution](https://128.84.21.199/abs/2003.03913)
*Zhanpeng Zhang, Kaipeng Zhang*
> (Submitted on 9 Mar 2020)

Real-time semantic segmentation is desirable in many robotic applications with limited computation resources. One challenge of semantic segmentation is to deal with the object scale variations and leverage the context. How to perform multi-scale context aggregation within limited computation budget is important. In this paper, firstly, we introduce a novel and efficient module called Cascaded Factorized Atrous Spatial Pyramid Pooling (CF-ASPP). It is a lightweight cascaded structure for Convolutional Neural Networks (CNNs) to efficiently leverage context information. On the other hand, for runtime efficiency, state-of-the-art methods will quickly decrease the spatial size of the inputs or feature maps in the early network stages. The final high-resolution result is usually obtained by non-parametric up-sampling operation (e.g. bilinear interpolation). Differently, we rethink this pipeline and treat it as a super-resolution process. We use optimized super-resolution operation in the up-sampling step and improve the accuracy, especially in sub-sampled input image scenario for real-time applications. By fusing the above two improvements, our methods provide better latency-accuracy trade-off than the other state-of-the-art methods. In particular, we achieve 68.4% mIoU at 84 fps on the Cityscapes test set with a single Nivida Titan X (Maxwell) GPU card. The proposed module can be plugged into any feature extraction CNN and benefits from the CNN structure development.


### [SuperMix: Supervising the Mixing Data Augmentation](https://128.84.21.199/abs/2003.05034)
*Ali Dabouei, Sobhan Soleymani, Fariborz Taherkhani, Nasser M. Nasrabadi*
>(Submitted on 10 Mar 2020)

In this paper, we propose a supervised mixing augmentation method, termed SuperMix, which exploits the knowledge of a teacher to mix images based on their salient regions. SuperMix optimizes a mixing objective that considers: i) forcing the class of input images to appear in the mixed image, ii) preserving the local structure of images, and iii) reducing the risk of suppressing important features. To make the mixing suitable for large-scale applications, we develop an optimization technique, 65× faster than gradient descent on the same problem. We validate the effectiveness of SuperMix through extensive evaluations and ablation studies on two tasks of object classification and knowledge distillation. On the classification task, SuperMix provides the same performance as the advanced augmentation methods, such as AutoAugment. On the distillation task, SuperMix sets a new state-of-the-art with a significantly simplified distillation method. Particularly, in six out of eight teacher-student setups from the same architectures, the students trained on the mixed data surpass their teachers with a notable margin.


### [Cars Can't Fly up in the Sky: Improving Urban-Scene Segmentation via Height-driven Attention Networks](https://128.84.21.199/abs/2003.05128)
*Sungha Choi, Joanne T. Kim, Jaegul Choo*
> (Submitted on 11 Mar 2020)

This paper exploits the intrinsic features of urban-scene images and proposes a general add-on module, called height-driven attention networks (HANet), for improving semantic segmentation for urban-scene images. It emphasizes informative features or classes selectively according to the vertical position of a pixel. The pixel-wise class distributions are significantly different from each other among horizontally segmented sections in the urban-scene images. Likewise, urban-scene images have their own distinct characteristics, but most semantic segmentation networks do not reflect such unique attributes in the architecture. The proposed network architecture incorporates the capability exploiting the attributes to handle the urban scene dataset effectively. We validate the consistent performance (mIoU) increase of various semantic segmentation models on two datasets when HANet is adopted. This extensive quantitative analysis demonstrates that adding our module to existing models is easy and cost-effective. Our method achieves a new state-of-the-art performance on the Cityscapes benchmark with a large margin among ResNet101 based segmentation models. Also, we show that the proposed model is coherent with the facts observed in the urban scene by visualizing and interpreting the attention map.


### [Rapid AI Development Cycle for the Coronavirus (COVID-19) Pandemic: Initial Results for Automated Detection & Patient Monitoring using Deep Learning CT Image Analysis](https://128.84.21.199/abs/2003.05037)
*Ophir Gozes, Maayan Frid-Adar, Hayit Greenspan, Patrick D. Browning, Adam Bernheim, Eliot Siegel*
>(Submitted on 10 Mar 2020 (v1), last revised 12 Mar 2020 (this version, v2))

Purpose: Develop AI-based automated CT image analysis tools for detection, quantification, and tracking of Coronavirus; demonstrate they can differentiate coronavirus patients from non-patients. Materials and Methods: Multiple international datasets, including from Chinese disease-infected areas were included. We present a system that utilizes robust 2D and 3D deep learning models, modifying and adapting existing AI models and combining them with clinical understanding. We conducted multiple retrospective experiments to analyze the performance of the system in the detection of suspected COVID-19 thoracic CT features and to evaluate evolution of the disease in each patient over time using a 3D volume review, generating a Corona score. The study includes a testing set of 157 international patients (China and U.S). Results: Classification results for Coronavirus vs Non-coronavirus cases per thoracic CT studies were 0.996 AUC (95%CI: 0.989-1.00) ; on datasets of Chinese control and infected patients. Possible working point: 98.2% sensitivity, 92.2% specificity. For time analysis of Coronavirus patients, the system output enables quantitative measurements for smaller opacities (volume, diameter) and visualization of the larger opacities in a slice-based heat map or a 3D volume display. Our suggested Corona score measures the progression of disease over time. Conclusion: This initial study, which is currently being expanded to a larger population, demonstrated that rapidly developed AI-based image analysis can achieve high accuracy in detection of Coronavirus as well as quantification and tracking of disease burden.


### [Highly Efficient Salient Object Detection with 100K Parameters](https://arxiv.org/abs/2003.05643)
*Shang-Hua Gao, Yong-Qiang Tan, Ming-Ming Cheng, Chengze Lu, Yunpeng Chen, Shuicheng Yan*
> (Submitted on 12 Mar 2020)

Salient object detection models often demand a considerable amount of computation cost to make precise prediction for each pixel, making them hardly applicable on low-power devices. In this paper, we aim to relieve the contradiction between computation cost and model performance by improving the network efficiency to a higher degree. We propose a flexible convolutional module, namely generalized OctConv (gOctConv), to efficiently utilize both in-stage and cross-stages multi-scale features, while reducing the representation redundancy by a novel dynamic weight decay scheme. The effective dynamic weight decay scheme stably boosts the sparsity of parameters during training, supports learnable number of channels for each scale in gOctConv, allowing 80% of parameters reduce with negligible performance drop. Utilizing gOctConv, we build an extremely light-weighted model, namely CSNet, which achieves comparable performance with about 0.2% parameters (100k) of large models on popular salient object detection benchmarks.

