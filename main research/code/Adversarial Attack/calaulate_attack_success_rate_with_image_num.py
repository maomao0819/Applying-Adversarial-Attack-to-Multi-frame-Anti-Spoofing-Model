import os
path_denominator = 'anti-spoofing test attack image'
path_numerator = 'adversarial_example_fgsm_range'
for dname in os.listdir(path_denominator):
    if os.path.isdir(os.path.join(path_denominator, dname)):
        image_num_denominator = 0
        image_num_numerator = 0
        for image in os.listdir(os.path.join(path_denominator, dname)):
            image_num_denominator += 1
        for image in os.listdir(os.path.join(path_numerator, dname)):
            image_num_numerator += 1
        with open(os.path.join(path_numerator, "attack success rate.txt"), 'a') as fp:
            fp.write(f"{dname} :  {str(image_num_numerator / image_num_denominator)}\n") 
        