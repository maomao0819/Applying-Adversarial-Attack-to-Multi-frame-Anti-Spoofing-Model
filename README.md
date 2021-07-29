# Applying-Adversarial-Attack-to-Multi-frame-Anti-Spoofing-Model
## 1.	Abstract
Face anti-spoofing is critical to the security of face recognition systems. Depth supervised learning has been proven as one of the most effective methods for face anti-spoofing. For security, utilizing a multi-frame model instead of single frame detection, and considering more frames can reduce the probability to be beaten. Since the recognition with multiple frames can not only check the patterns in the frame but also ensure the time relative relationship between different frames, the multi-frame anti-spoofing model seems unbeatable. In this paper, we study the vulnerability of anti-spoofing methods against adversarial perturbations, and try to defeat the multi-frame model. To conduct the experiment, we adapt the basic adversarial attack method to attack on different dataset. Furthermore, we introduce the concept of image kernel filter to attempt increasing attack success rate and optimizing image quality.
## 2. Introduction
Face recognition technology has become mature, the detection can precisely tell who you are. Nonetheless, some people can use your photo to pretend to be you and use your identity to do something harmful to you. Thus, the face anti-spoofing technology is not less important than recognition. To optimize the detection system, many researchers consider more filters as the one of the best solutions. Nowadays, more and more face anti-spoofing methods on multiple frames have been designed, and there are many attack algorithms on images. However, there is no paper about how the performance is about attacking on the multi-frame face anti-spoofing model.  We wonder if the multi-frame model is vulnerable to adversarial attack, and other researchers may improve the model by erasing the weakness. And to increase the performance of attacking, we introduce the concept of the kernel image filters and SSIM to improve the image quality and the attack success rate on multi-frame face anti-spoofing model.
## 3. Related Work
We will firstly train a model which can precisely classify the spoofing images and real images with multiple frames. We adapted the detection model [1] for our work, and the model considers 1) detailed discriminative clues between living and spoofing faces may be discarded through stacked vanilla convolutions, and 2) the dynamics of 3D moving faces provide important clues in detecting the spoofing faces. Furthermore, since the model is combined of Binary supervised Methods, Depth supervised Methods, and Temporal-based Methods, we believe it is powerful enough for our attacking experiment. And then, we try to beat the model with the basic adversarial attack method such as FGSM and iterative FGSM. They are both attack methods which consider the gradient of the error.

Several attack methods have been proposed to generate adversarial examples. Here we provide a brief introduction.

Fast Gradient Sign Method (FGSM). FGSM [2] (Goodfellow et al., 2014) generates an adversarial example xadv by maximizing the loss function J(x<sup>adv</sup>, y<sup>true</sup>) with one-step update as:

x<sup>adv</sup> = x + eps * sign(∇<sub>x</sub>J(x, y<sup>true</sup>)),

where sign(·) function restricts the perturbation in the L∞ norm bound.  

[3] Iterative Fast Gradient Sign Method (I-FGSM). Kurakin et al. (2016) extend FGSM to an iterative version by applying FGSM with a very small step size α: 

x<sub>0</sub> = x, x<sub>t+1</sub><sup>adv</sup> = Clip<sub>x</sub><sup>eps</sup>{x<sub>t</sub><sup>adv</sup> + α · sign(∇xJ(xt<sup>adv</sup> , y<sup>true</sup>))},

where Clip<sub>x</sub><sup>eps</sup> (·) function restricts generated adversarial examples to be within the eps-ball of x.
## 4. Methodology:
In the beginning, we train a model which can precisely tell the spoofing images and real images with multi-frames. We adapted the detection model [1] for our work and we believe it is powerful enough for our attacking experiment. And then, we try to use adversarial attack like FGSM and iFGSM with kernel image filters to beat the model.

What with the attack success rate and what with the image quality, we introduce image kernel filters. Since the model is temporal-based method, the filter concept may increase the fool rate. To be more specific, the model considers the relationship between 5 frames, so, it is not wise to add the perturbation without considering the temporal relationship. Unfortunately, the adversarial attack may break the relationship severely. Since the attack will lead the pixel on different frames to be so much different, it may cause the image with the noise to remain being spoofing. Remember, the goal of the attack is to let the model misclassify the images. To alleviate the impact on temporality, we use the filter to reduce the difference between the pixels on different frames. Consequently, We try the basic and common filter on our experiment.

Because we hope the images with noise are natural, and people shouldn’t get weird from the images with perturbation, we pursue images to be high quality. For measuring the quality, we introduce the SSIM index to demonstrate whether the image is good or bad. Besides, we attempt to add multiple filters and hope it will turn out to be a better result of attack success rate and image quality.
## 5.	Experiment
### 5.1. Attacked target
Since our goal is to fool the detecting recognition, we only attack the image whose label is spoofing and the model also correctly classifies it as a spoofing image.
### 5.2. Dataset
Four datasets - OULU-NPU, MSU-MFSD, CASIA-FASD, Replay-Attack are used in our experiment. OULU-NPU is a high-resolution database,consisting of 4950 real access and spoofing videos and containing four protocols to validate the generalization of models. MSU-MFSD contains 280 video recordings of genuine and attack faces. CASIA-FASD and Replay-Attack both contain low-resolution videos.
### 5.3. Performance Metrics
# ASR: Attack Success Rate, which is the ratio represented by the spoofing images misclassified to be the real by the model.
# SSIM: Structural Similarity Index Measure, which is the index that elaborates how two images are similar by considering structure, intensity, and contrast. 
### 5.4. Comparison
#### 5.4.1. model
![](https://user-images.githubusercontent.com/43957213/127464687-4a9c3ed7-c031-405c-8f08-6e2647249a53.jpg)
The model is strong enough to classify real or spoofing, and it is worth being attacked for our work. 
#### 5.4.2. FGSM vs iFGSM	
The table show the attack success rate (%) with different epsilon.
![](https://user-images.githubusercontent.com/43957213/127465317-8c0ae782-5c70-4e22-9afe-c9302a5c0d71.jpg)
## 6.	Demo System:
* Step 1.  Film a video with cellphone(about 1 min)
* Step 2.  Refilm the video with the camera of laptop
* Step 3.  Cut the video in step 2. into frames
* Step 4.  Crop face to every frames
* Step 5.  Generate depth map
* Step 6.  Go through our multi-frame model to check whether the input frames are spoofing to model 
* Step 7.  Attack by FGSM (+filter) and Iterative FGSM (+filter)\
## 7.	Conclusion:
In conclusion, it is more difficult to fool the multi-frame model than the single frame model. If we follow the previous Fast Gradient Sign Method, although we can make each frame closer to the real image respectively, it may cause the difference between frames to become larger, and cause the model to still judge it as a spoof image. In our experiment, we proposed a novel method. We add a smoothing filter to the perturbation generated by Fast Gradient Sign Method, which can make the mask we want to add on our frame become more uniform. This allows us to reduce the difference between frames as much as possible while bringing them closer to the real image. Finally, making it easier for the model to judge it as a real image. Extensive experiments demonstrate the superiority of our method.

And, the followings show the brief conclusion of the experiment result, and the attack is on multi-frame model. 
* Filter will enhance the effect of FGSM.
* Filter will worsen the performance of iFGSM
*	Filter Performance：Gaussian > Uniform
* SSIM: Gaussian > Uniform > no filter
* Multiple filter increase image quality
* Best filter’s sigma is around 1
* The best number of the layers is only 1.
* Trade-off between ASR and SSIM is important
* Best attack: iFGSM or FGSM with Filter 

## 8. Reference:
[1] WANG, Zezheng, et al. Deep spatial gradient and temporal depth learning for face anti-spoofing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020. p. 5042-5051.

[2] GOODFELLOW, Ian J.; SHLENS, Jonathon; SZEGEDY, Christian. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014

[3] KURAKIN, Alexey, et al. Adversarial examples in the physical world. 2016.
