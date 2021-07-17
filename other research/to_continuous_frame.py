import os
import shutil
import pathlib

root = '/mnt/sda1/mike/OULU-NPU'
train_feature_src = os.path.join(root, 'Train_features')
train_depth_feature_src = os.path.join(root, 'Train_Depth_features')
dev_feature_src = os.path.join(root, 'Dev_features')
dev_depth_feature_src = os.path.join(root, 'Dev_Depth_features')
test_feature_src = os.path.join(root, 'Test_features')
test_depth_feature_src = os.path.join(root, 'Test_Depth_features')

train_feature_dst = os.path.join(root, 'Train_features_cont_frame_nested')
train_depth_feature_dst = os.path.join(root, 'Train_Depth_features_cont_frame_nested')
dev_feature_dst = os.path.join(root, 'Dev_features_cont_frame_nested')
dev_depth_feature_dst = os.path.join(root, 'Dev_Depth_features_cont_frame_nested')
test_feature_dst = os.path.join(root, 'Test_features_cont_frame_nested')
test_depth_feature_dst = os.path.join(root, 'Test_Depth_features_cont_frame_nested')

feature_path_src_list = [train_feature_src, dev_feature_src, test_feature_src]
depth_path_src_list = [train_depth_feature_src, dev_depth_feature_src, test_depth_feature_src]
feature_path_dst_list = [train_feature_dst, dev_feature_dst, test_feature_dst]
depth_path_dst_list = [train_depth_feature_dst, dev_depth_feature_dst, test_depth_feature_dst]

def CheckDirectory(path):
    if not os.path.exists(path):
        pathlib.Path(path).mkdir()

for i in range(len(feature_path_dst_list)):
	CheckDirectory(feature_path_dst_list[i])
	CheckDirectory(depth_path_dst_list[i])

extend1 = '.png'
extend2 = '.jpg'

for i in range(len(feature_path_src_list)):
	files = sorted(os.listdir(feature_path_src_list[i]))
	file_list = []
	file_list.append(files[0])
	files.remove(files[0])
	video_id = file_list[-1].split('-')[0]
	for file in files:
		if file.split('-')[0] != video_id:
			frame_id = 1
			file_list = sorted(file_list, key = lambda f: int(f.split('.')[0].split('-')[-1]))
			for f in file_list:
				# f_dst = f.split('-')[0] + '-' + str(frame_id) + extend1
				# frame_id += 10
				# shutil.copy2(os.path.join(feature_path_src_list[i], f), os.path.join(feature_path_dst_list[i], f_dst))
				CheckDirectory(os.path.join(feature_path_dst_list[i], video_id))
				shutil.copy2(os.path.join(feature_path_src_list[i], f), os.path.join(feature_path_dst_list[i], video_id, str(frame_id) + extend1))
				frame_id += 10
			video_id = file.split('-')[0]
			file_list.clear()
		file_list.append(file)

	files = sorted(os.listdir(depth_path_src_list[i]))
	file_list = []
	file_list.append(files[0])
	files.remove(files[0])
	video_id = file_list[-1].split('-')[0]
	for file in files:
		if file.split('-')[0] != video_id:
			frame_id = 1
			file_list = sorted(file_list, key = lambda f: int(f.split('-')[-1].split('_')[0]))
			for f in file_list:
				# f_dst = f.split('-')[0] + '-' + str(frame_id) + '_depth' + extend2
				# frame_id += 10
				# shutil.copy2(os.path.join(depth_path_src_list[i], f), os.path.join(depth_path_dst_list[i], f_dst))
				CheckDirectory(os.path.join(depth_path_dst_list[i], video_id))
				shutil.copy2(os.path.join(depth_path_src_list[i], f), os.path.join(depth_path_dst_list[i], video_id, str(frame_id) + '_depth' + extend2))
				frame_id += 10
			video_id = file.split('-')[0]
			file_list.clear()
		file_list.append(file)
	

	

