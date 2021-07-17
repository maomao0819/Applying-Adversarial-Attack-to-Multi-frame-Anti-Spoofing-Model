import os
import pathlib
import shutil
target_path = os.path.join('homemade_test', 'outputs_PSNR_fgsm')
find_path = os.path.join('adversarial_example_ifgsm_range_with_PSNR', 'homemade_test')
path_save = os.path.join('homemade_test', 'outputs_PSNR_ifgsm')
def CheckDirectory(path_save):
	if not os.path.exists(path_save):
		pathlib.Path(path_save).mkdir()
if __name__ == "__main__":
	CheckDirectory(path_save)
	image_idx = []
	for image in os.listdir(target_path):
		image_idx.append(image.split('_')[2])
	for image in os.listdir(find_path):
		if image.split('_')[2] in image_idx:
			shutil.copyfile(os.path.join(find_path, image), os.path.join(path_save, image)) 
