import os
import numpy as np

path = 'adversarial_example_ifgsm_eps_0.05_or_0.06'
total_image_num = 0
total_attack_success = 0

with open(os.path.join(path, "attack success rate.txt")) as fp: 
	Lines = fp.readlines()
	for line in Lines:
		image_num = 0
		line_content = line.split()
		dname = line_content[0]
		attack_success_rate = line_content[-1]
		if os.path.isdir(os.path.join(path, dname)):
			for i in os.listdir(os.path.join(path, dname)):
				image_num += 1
		total_attack_success += image_num * np.float32(attack_success_rate)
		total_image_num += image_num

f = open(os.path.join(path, "attack success rate.txt"), "a")
f.write(f"total :  {str(total_attack_success / total_image_num)}\n")
f.close()