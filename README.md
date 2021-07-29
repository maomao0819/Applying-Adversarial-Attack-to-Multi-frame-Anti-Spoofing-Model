# Applying-Adversarial-Attack-to-Multi-frame-Anti-Spoofing-Model
## 1.	Abstract
Face anti-spoofing is critical to the security of face recognition systems. Depth supervised learning has been proven as one of the most effective methods for face anti-spoofing. For security, utilizing a multi-frame model instead of single frame detection, and considering more frames can reduce the probability to be beaten. Since the recognition with multiple frames can not only check the patterns in the frame but also ensure the time relative relationship between different frames, the multi-frame anti-spoofing model seems unbeatable. **In this paper, we study the vulnerability of anti-spoofing methods against adversarial perturbations, and try to defeat the multi-frame model.** To conduct the experiment, we adapt the basic adversarial attack method to attack on different dataset. Furthermore, we introduce the concept of image kernel filter to attempt increasing attack success rate and optimizing image quality.
## 2. Introduction
Face recognition technology has become mature, the detection can precisely tell who you are. Nonetheless, some people can use your photo to pretend to be you and use your identity to do something harmful to you. Thus, the face anti-spoofing technology is not less important than recognition. To optimize the detection system, many researchers consider more filters as the one of the best solutions. Nowadays, more and more face anti-spoofing methods on multiple frames have been designed, and there are many attack algorithms on images. However, there is no paper about how the performance is about attacking on the multi-frame face anti-spoofing model.  We wonder if the multi-frame model is vulnerable to adversarial attack, and other researchers may improve the model by erasing the weakness. And to increase the performance of attacking, **we introduce the concept of the kernel image filters and SSIM to improve the image quality and the attack success rate on multi-frame face anti-spoofing model.**
## 3. Related Work
We will firstly train a model which can precisely classify the spoofing images and real images with **multiple frames**. We adapted the detection model [1] for our work, and the model considers 1) detailed discriminative clues between living and spoofing faces may be discarded through stacked vanilla convolutions, and 2) the dynamics of 3D moving faces provide important clues in detecting the spoofing faces. Furthermore, since the model is combined of Binary supervised Methods, Depth supervised Methods, and Temporal-based Methods, we believe it is powerful enough for our attacking experiment. And then, we try to beat the model with the basic adversarial attack method such as FGSM and iterative FGSM. They are both attack methods which consider the gradient of the error.

Several attack methods have been proposed to generate adversarial examples. Here we provide a brief introduction.

**Fast Gradient Sign Method (FGSM).** FGSM [2] (Goodfellow et al., 2014) generates an adversarial example xadv by maximizing the loss function J(x<sup>adv</sup>, y<sup>true</sup>) with one-step update as:

x<sup>adv</sup> = x + eps * sign(∇<sub>x</sub>J(x, y<sup>true</sup>)),

where sign(·) function restricts the perturbation in the L∞ norm bound.  

[3] **Iterative Fast Gradient Sign Method (I-FGSM).** Kurakin et al. (2016) extend FGSM to an iterative version by applying FGSM with a very small step size α: 

x<sub>0</sub> = x, x<sub>t+1</sub><sup>adv</sup> = Clip<sub>x</sub><sup>eps</sup>{x<sub>t</sub><sup>adv</sup> + α · sign(∇xJ(x<sub>t</sub><sup>adv</sup> , y<sup>true</sup>))},

where Clip<sub>x</sub><sup>eps</sup> (·) function restricts generated adversarial examples to be within the eps-ball of x.
## 4. Methodology:
In the beginning, we train a model which can precisely tell the spoofing images and real images with multi-frames. We adapted the detection model [1] for our work and we believe it is powerful enough for our attacking experiment. And then, we try to use adversarial attack like FGSM and iFGSM with kernel image filters to beat the model.

What with the attack success rate and what with the image quality, we introduce image kernel filters. **Since the model is temporal-based method, the filter concept may increase the fool rate.** To be more specific, the model considers the relationship between 5 frames, so, it is not wise to add the perturbation without considering the temporal relationship. Unfortunately, the adversarial attack may break the relationship severely. Since the attack will lead the pixel on different frames to be so much different, it may cause the image with the noise to remain being spoofing. Remember, the goal of the attack is to let the model misclassify the images. To alleviate the impact on temporality, **we use the filter to reduce the difference between the pixels on different frames.** Consequently, We try the basic and common filter on our experiment. Finally, we adopt the average filter [4] and Gaussian filter to improve the image quality and attack success rate. **Above all, it may be the first paper to propose the filter to increase the image quality and the attack success rate on multi-frame face anti-spoofing.**

Because we hope the images with noise are natural, and people shouldn’t get weird from the images with perturbation, we pursue images to be high quality. **For measuring the quality, we introduce the SSIM index** to demonstrate whether the image is good or bad. Besides, we attempt to add multiple filters and hope it will turn out to be a better result of attack success rate and image quality.
## 5.	Experiment
### 5.1. Attacked target
Since our goal is to fool the detecting recognition, we only attack the image whose label is spoofing and the model also correctly classifies it as a spoofing image.
### 5.2. Dataset
Four datasets - OULU-NPU, MSU-MFSD, CASIA-FASD, Replay-Attack are used in our experiment. OULU-NPU is a high-resolution database,consisting of 4950 real access and spoofing videos and containing four protocols to validate the generalization of models. MSU-MFSD contains 280 video recordings of genuine and attack faces. CASIA-FASD and Replay-Attack both contain low-resolution videos.
### 5.3. Performance Metrics
* ASR: Attack Success Rate, which is the ratio represented by the spoofing images misclassified to be the real by the model.
* SSIM: Structural Similarity Index Measure, which is the index that elaborates how two images are similar by considering structure, intensity, and contrast. 
### 5.4. Comparison
#### 5.4.1. model
![](https://user-images.githubusercontent.com/43957213/127464687-4a9c3ed7-c031-405c-8f08-6e2647249a53.jpg)

The model is strong enough to classify real or spoofing, and it is worth being attacked for our work. 
#### 5.4.2. FGSM vs iFGSM	
The table show the attack success rate (%) with different epsilon.
![](https://user-images.githubusercontent.com/43957213/127465317-8c0ae782-5c70-4e22-9afe-c9302a5c0d71.jpg)

The bottom-left corner value is the attack success rate / SSIM.
![](https://user-images.githubusercontent.com/43957213/127465499-378935fa-9786-4d86-9a5f-0f1995b6a1b6.png)
For the dataset OULU-1, the result of the iFGSM is better than FGSM. However, for the others datasets, the preformance of the iFGSM is worse than FGSM.
#### 5.4.3. the filter, the dimension ,and the application place 
We attempt to apply the concept of the image filter and try to find where is the best place to apply the kernel filter and which filter is hitting the point. To do the appropriate comparison, we have made some experiment. The table show that which filter is applied, applied on 2-dimension or 3-dimention to the perturbation and where does the filter apply. For the filter type, we only apply Gaussian filter and Uniform filter. Furthermore,‘noise’means we apply the filter to the perturbation,‘image’means we only apply the filter to the image after adding perturbation, and ‘all’means we apply the filter to the perturbation and the image after adding perturbation.

The table will show the attack success rate (%) with the result of the different eps and auxiliary filters based on the same dataset and the same adversarial attack method. 

The dataset is **OULU-1**, and the attack method is **FGSM**.
![](https://user-images.githubusercontent.com/43957213/127465718-bb0d9922-2d6d-4bd1-bdd9-93328cc81457.jpg)

The dataset is **OULU-1**, and the attack method is **iFGSM**.
![](https://user-images.githubusercontent.com/43957213/127465932-64679603-2398-4c11-ac62-4c95ffe8cad8.jpg)

The dataset is **OULU-2**, and the attack method is **FGSM**.
![](https://user-images.githubusercontent.com/43957213/127466189-0e90bab2-d216-4b2c-972d-89026178c584.jpg)

The dataset is **OULU-1**, and the attack method is **iFGSM**.
![](https://user-images.githubusercontent.com/43957213/127466293-29e4ab95-a7db-4c07-962a-1d1f0eda5734.jpg)

The bottom-left corner value is the attack success rate.
![](https://user-images.githubusercontent.com/43957213/127466573-9bd37412-e4cf-4c84-bf26-602733e08895.png)
![](https://user-images.githubusercontent.com/43957213/127466662-ae951eed-0ddd-4d32-8639-7aa50eb5528d.png)
![](https://user-images.githubusercontent.com/43957213/127466683-baf67e70-fe06-4e69-999a-3caacb726b6f.png)
The result on other dataset is close. In conclusion, **For FGSM, the fittest method is applying 3D Gaussian filter to the perturbation of FGSM,** and we can significantly fool the multi-frame anti-spoofing model. However, **for iFGSM, the auxiliary filters doesn’t brings any advantage** to the attack, and they do nothing but reduce the attack success rate. Thus, the auxiliary image filters only work on FGSM.

We also conduct other experiments with PIL smooth and smooth_more filters. Unfortunately, the performance of those isn’t good enough. 
#### 5.4.4.	SSIM, Sigma and Multiple filter on 3D Gaussian and FGSM
Since the filter doesn’t work on iFGSM and the Gaussian filter is better than the uniform filter, we only apply 3D Gaussian filter to the perturbation of FGSM to the rest of the experiment. For the sake of not changing the image so severely, we introduce the SSIM to indicate the image texture. Moreover, we conduct more experiment on the filter. We test different parameters (sigma) of the Gaussian filter and try to apply more layer of the filter. 

The table show the attack success rate (%) and the SSIM with the result of the different eps, sigma, the number of the layers.

The dataset is **OULU-1**, and the attack method is **FGSM**.
![](https://user-images.githubusercontent.com/43957213/127466992-6e84f9c1-044e-45fe-86ce-4442517ec124.jpg)

The dataset is **OULU-2**, and the attack method is **FGSM**.
![](https://user-images.githubusercontent.com/43957213/127467057-d8d35e16-d7d2-4323-940a-a3474d12efac.jpg)

The sigma is fixed on 1 and the bottom-left corner value is the attack success rate / SSIM, and it tell the different images of different numbers of the Gaussian layers.
![](https://user-images.githubusercontent.com/43957213/127467163-16cac790-87f3-4cc6-a3f9-6843a47a354b.png)
The bottom-left corner value is the attack success rate | SSIM.
![](https://user-images.githubusercontent.com/43957213/127467233-91c38409-1a49-4612-ada8-8873bba65c39.png)
In conclusion, **the most proper sigma is around 1.** In addition, although 2 layers of the filter can increase the image quality, it losts lots of attack success rate. After considering the trade-off, we regard the multiple layer method as a not good Auxiliary tools.
#### 5.4.5. Comprehension Comparison
The ASR of the nearly image quality on two protocols of OULU-NPU
![](https://user-images.githubusercontent.com/43957213/127467369-83468ca3-0b58-4758-aed0-e5f8c79524dc.jpg)

The bottom-left corner value is the attack success rate.
![](https://user-images.githubusercontent.com/43957213/127467436-3054ee08-1ca8-4c5f-8531-ff34dafc3d60.png)
**For dataset OULU-1, the performance of FGSM + filter is the best, and for dataset OULU-2, the result of iFGSM is most significant** among all. Furthermore, **filter can improve FGSM performance.**
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
* **Filter will enhance the effect of FGSM.**
* **Filter will worsen the performance of iFGSM.**
*	**Filter Performance：Gaussian > Uniform.**
* **SSIM: Gaussian > Uniform > no filter.**
* **Multiple filter increase image quality.**
* **Best filter’s sigma is around 1.**
* **The best number of the layers is only 1.**
* **Trade-off between ASR and SSIM is important.**
* **Best attack: iFGSM or FGSM with Filter.**

## 8. Reference:
[1] WANG, Zezheng, et al. Deep spatial gradient and temporal depth learning for face anti-spoofing. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020. p. 5042-5051.

[2] GOODFELLOW, Ian J.; SHLENS, Jonathon; SZEGEDY, Christian. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014

[3] KURAKIN, Alexey, et al. Adversarial examples in the physical world. 2016.

[4] FEIBUSH, Eliot A.; LEVOY, Marc; COOK, Robert L. Synthetic texturing using digital filters. In: Proceedings of the 7th annual conference on Computer graphics and interactive techniques. 1980. p. 294-301.

[5] BERGHOLM, Fredrik. Edge focusing. IEEE transactions on pattern analysis and machine intelligence, 1987, 6: 726-741.
