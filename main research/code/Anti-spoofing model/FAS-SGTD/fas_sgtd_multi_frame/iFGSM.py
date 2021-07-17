from easydict import EasyDict as edict
import numpy as np
from matplotlib import pyplot as plt
import os
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
# tf.disable_v2_behavior()
from generate_data_test_without_blank_depth import input_fn_maker
import FLAGS_tf2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import uniform_filter

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test_config = tf.ConfigProto(device_count={"GPU": 0}, allow_soft_placement=True, log_device_placement=True)
test_config.gpu_options.allow_growth = True

flags = FLAGS_tf2.flags
eps = 0.4

loss_type = "CEP"
if loss_type == "CEP":
    from generate_network import generate_network as model_fn
if loss_type == "paper":
    from generate_network_other_loss import generate_network as model_fn

logit_type = "normal"
# paper
#     logits[0] = depth_ratio * depth_mean + cla_ratio * logits[0]
#     logits[1] = 1.0 - depth_mean
# normal
#     logits[0]
#     logits[1]
create_original_image = False
filter_type = flags.paras.filter_type
filter_dimension = flags.paras.filter_dimension
filter_sigma = flags.paras.filter_sigma
filter_mode = flags.paras.filter_mode
# test_file=['/mnt/sdb1/mike/Datasets/OULU-NPU/maomao_test/feature', '/mnt/sdb1/mike/Datasets/OULU-NPU/maomao_test/depth']
# dataset = 'OULU-NPU'
# dataset_path = os.path.join('/', 'mnt', 'sdb1', 'mike', 'Datasets', dataset)
# test_file = [os.path.join(dataset_path, 'mtcnn', 'Test_features'), os.path.join(dataset_path, 'Test_Depth_all')]
test_file = flags.path.test_file
test_data_list = [test_file]


eval_input_fn = lambda: input_fn_maker(test_data_list, shuffle=False, batch_size=1, epoch=1)


data = input_fn_maker(test_data_list, shuffle=False, batch_size=1, epoch=1)


# print((data['images'].shape))
# print(type(data['images']))
# print((data['maps'].shape))
# print(type(data['maps']))
# print((data['masks'].shape))
# print(type(data['masks']))
# print((data['labels'].shape))
# print(type(data['labels']))


# latest_ckp = tf.train.latest_checkpoint('model.ckpt-19501')
# checkpoint_path = 'model_github_backup/model.ckpt-19501'
# checkpoint_path = './model_multipler_2_inverse_19501/OULU-NPU_1/model.ckpt-25001'
# checkpoint_path = "./model_multipler_2_inverse/replayattack/model.ckpt-20000"
checkpoint_path = "./model_multipler_2_inverse_19501/OULU-NPU_2/model.ckpt-97501"
# checkpoint_path = './model_multipler_2_inverse_19501/OULU-NPU_3/model.ckpt-100001'
# checkpoint_path = "./model_multipler_2_inverse_19501/MSU-MFSD/model.ckpt-100001"
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name="", all_tensors=False, all_tensor_names=True)


# GPU config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

# sess = tf.compat.v1.Session(config=config)

# create estimator
this_config = tf.estimator.RunConfig(
    save_summary_steps=10000000000,
    save_checkpoints_steps=None,
    keep_checkpoint_max=1024,
    log_step_count_steps=None,
    session_config=config,
)

# model fn
model_fn_this = model_fn

# model_dir
# inference_model_dir = 'model_github_backup/'
# inference_model_dir = './model_multipler_2_inverse_19501/OULU-NPU_1/'
# inference_model_dir = "./model_multipler_2_inverse/replayattack"
inference_model_dir = "./model_multipler_2_inverse_19501/OULU-NPU_2/"
# inference_model_dir = './model_multipler_2_inverse_19501/OULU-NPU_3/'
# inference_model_dir = "./model_multipler_2_inverse_19501/MSU-MFSD/"
mnist_classifier = tf.estimator.Estimator(model_fn=model_fn_this, config=this_config, model_dir=inference_model_dir)


save_path = "./saved_model/"

feature_spec = {
    "images": tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 15], name="images"),
    "maps": tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 5], name="maps"),
    "masks": tf.placeholder(dtype=tf.float32, shape=[None, 256, 256, 15], name="masks"),
    "labels": tf.placeholder(dtype=tf.int32, shape=[None, 1], name="labels"),
    "names": tf.placeholder(dtype=tf.int32, shape=[1], name="names"),
    "folder": tf.placeholder(dtype=tf.string, shape=[1], name="folder"),
}

input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)

export_model_path = mnist_classifier.export_savedmodel(save_path, input_receiver_fn)
export_model_path = export_model_path.decode()


# print(type(data["images"]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    images_np = data["images"].eval()

# print(type(images_np))


features = mnist_classifier.predict(input_fn=eval_input_fn, checkpoint_path=checkpoint_path)


features_list = list(features)


n_attack_success = 0
n_test_real = 0
if not os.path.exists("iFGSM_result"):
    os.makedirs("iFGSM_result")

dataset = flags.path.dataset
if dataset == "OULU-NPU":
    result_path = os.path.join(
        "iFGSM_result",
        dataset,
        flags.dataset.protocal,
        f"logit_type_{logit_type}",
        f"loss_type_{loss_type}",
        f"eps_{eps}",
    )
    if filter_type == 1:
        if filter_sigma > 0:
            result_path = os.path.join(
                "iFGSM_result",
                dataset,
                flags.dataset.protocal,
                f"logit_type_{logit_type}",
                f"loss_type_{loss_type}",
                f"gaussian_filter_{filter_dimension}D",
                f"filter_mode_{str(filter_mode)}",
                f"filter_sigma_{str(filter_sigma)}",
                f"eps_{eps}",
            )
    elif filter_type == 2:
        result_path = os.path.join(
            "iFGSM_result",
            dataset,
            flags.dataset.protocal,
            f"logit_type_{logit_type}",
            f"loss_type_{loss_type}",
            f"uniform_filter_{filter_dimension}D",
            f"filter_mode_{str(filter_mode)}",
            f"eps_{eps}",
        )
else:
    result_path = os.path.join(
        "iFGSM_result", dataset, f"logit_type_{logit_type}", f"loss_type_{loss_type}", f"eps_{eps}",
    )
    if filter_type == 1:
        if filter_sigma > 0:
            result_path = os.path.join(
                "iFGSM_result",
                dataset,
                f"logit_type_{logit_type}",
                f"loss_type_{loss_type}",
                f"gaussian_filter_{filter_dimension}D",
                f"filter_mode_{str(filter_mode)}",
                f"filter_sigma_{str(filter_sigma)}",
                f"eps_{eps}",
            )
    elif filter_type == 2:
        result_path = os.path.join(
            "iFGSM_result",
            dataset,
            f"logit_type_{logit_type}",
            f"loss_type_{loss_type}",
            f"uniform_filter_{filter_dimension}D",
            f"filter_mode_{str(filter_mode)}",
            f"eps_{eps}",
        )
if not os.path.exists(result_path):
    os.makedirs(result_path)

if flags.paras.inverse:
    label_spoof = 1
    label_real = 0
else:
    label_spoof = 0
    label_real = 1

if eps > 1:
    iteration_time_N = min(eps + 4, 1.25 * eps)
elif eps > 0:
    iteration_time_N = min(eps * 255 + 4, 1.25 * eps * 255)
    iteration_time_N = 4
else:
    iteration_time_N = 0
int_iteration_time_N = np.round(iteration_time_N, decimals=0).astype(np.int32)
if int_iteration_time_N:
    # weight_alpha = np.round(eps / iteration_time_N if (eps / iteration_time_N) > 1 else 1)
    weight_alpha = eps / iteration_time_N


def clip(origin_image, adv_x, eps):
    # origin_image = np.array((origin_image * 0.5 + 0.5) * 255) #-1 ~ 1 to 0 ~ 255
    # adv_x = np.array((adv_x * 0.5 + 0.5) * 255)
    # eps *= 255
    for x in range(adv_x.shape[1]):
        for y in range(adv_x.shape[2]):
            for z in range(adv_x.shape[3]):
                # adv_x[0, x, y, z] = min(255, origin_image[0, x, y, z] + eps, max(0, origin_image[0, x, y, z] - eps, adv_x[0, x, y, z]))
                adv_x[0, x, y, z] = min(
                    1, origin_image[0, x, y, z] + eps, max(0, origin_image[0, x, y, z] - eps, adv_x[0, x, y, z]),
                )
                # return tf.convert_to_tensor((adv_x / 255 - 0.5) * 2)
    return adv_x / 2


for folder_idx in range(len(features_list)):
    feature = features_list[folder_idx]
    logits = feature["logits"]
    folder = feature["folder"][0].decode()
    folder_names = feature["names"][0].decode()
    if create_original_image:
        images_squeeze = np.squeeze(np.expand_dims(feature["images"], axis=0))
        frames = np.split(images_squeeze, indices_or_sections=5, axis=2)
        if dataset == "OULU-NPU":
            if not os.path.exists(f"original_image/{dataset}/{flags.dataset.protocal}/{folder_names}"):
                os.makedirs(f"original_image/{dataset}/{flags.dataset.protocal}/{folder_names}")
        else:
            if not os.path.exists(f"original_image/{dataset}/{flags.dataset.protocal}/{folder}/{folder_names}"):
                os.makedirs(f"original_image/{dataset}/{flags.dataset.protocal}/{folder}/{folder_names}")
        for frame_id in range(5):
            img = np.clip(frames[frame_id] + 0.5, 0, 1)
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
            if dataset == "OULU-NPU":
                img.save(f"original_image/{dataset}/{flags.dataset.protocal}/{folder_names}/frame_{frame_id + 1}.png")
            else:
                img.save(
                    f"original_image/{dataset}/{flags.dataset.protocal}/{folder}/{folder_names}/frame_{frame_id + 1}.png"
                )

    if loss_type == "paper":
        maps_feature = feature["maps"]
        maps_feature = maps_feature[np.newaxis, :]
    if logit_type == "paper":
        depth_map = feature["depth_map"]
        masks = feature["masks"]
        depth_map = depth_map[..., 0] * masks[..., 0]
        depth_mean = np.sum(depth_map) / np.sum(masks[..., 0])
        cla_ratio = flags.paras.cla_ratio
        depth_ratio = 1 - cla_ratio
        logits[label_real] = depth_ratio * depth_mean + cla_ratio * logits[label_real]
        logits[label_spoof] = 1.0 - depth_mean
    folder_names = feature["names"][0].decode()
    is_test_real = logits[label_real] > logits[label_spoof]
    if is_test_real:
        with open(os.path.join(result_path, "logits.txt"), "a+") as f:
            if dataset == "OULU-NPU":
                f.write(f"folder: {folder_names}\toriginal: {logits}\ttest real\n")
            else:
                f.write(f"folder: {folder}/{folder_names}\toriginal: {logits}\ttest real\n")
        n_test_real += 1
        continue
    label = feature["labels"]
    # print(label.shape)
    # print(label)

    # if label == 0:
    #     label = 1
    # else:
    #     label = 0
    label = 1 if label == 0 else 0

    labels_onehot = np.eye(2)[label]
    # print(logits)
    # print(label)
    # print(labels_onehot)

    label = np.array(label)
    # print('label', label)
    label = np.reshape(label, (1, 1))
    # print(label.shape)

    # 1
    # model_path = './saved_model/1618330861'
    # 2
    model_path = export_model_path
    # all spoof
    # model_path = './saved_model/1618332474'
    images_np = np.expand_dims(feature["images"], axis=0)
    adv_images = images_np
    for i in range(int_iteration_time_N + 1):
        with tf.Session(graph=tf.Graph()) as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
            signature = meta_graph_def.signature_def
            # print(signature)
            x_tensor_name = signature["serving_default"].inputs["images"].name
            label_tensor_name = signature["serving_default"].inputs["labels"].name
            if loss_type == "paper":
                maps_tensor_name = signature["serving_default"].inputs["maps"].name
            # print('x_tensor_name', x_tensor_name)
            logit_tensor_name = signature["serving_default"].outputs["logits"].name
            y_tensor_name = signature["serving_default"].outputs["grads"].name
            # print('y_tensor_name', y_tensor_name)
            x = sess.graph.get_tensor_by_name(x_tensor_name)
            y = sess.graph.get_tensor_by_name(y_tensor_name)
            if loss_type == "paper":
                maps = sess.graph.get_tensor_by_name(maps_tensor_name)
            labels = sess.graph.get_tensor_by_name(label_tensor_name)
            logits = sess.graph.get_tensor_by_name(logit_tensor_name)
            # print('images_np', images_np.shape)
            # print('label', label.shape)
            # print('maps', maps_feature.shape)
            images_np = np.expand_dims(feature["images"], axis=0)
            logit = sess.run(logits, {x: images_np, labels: label})
            if loss_type == "paper":
                grads = sess.run(y, {x: images_np, labels: label, maps: maps_feature})
            elif loss_type == "CEP":
                grads = sess.run(y, {x: images_np, labels: label})
            # print(grads)
            # print(grads.shape)

            # print(predictions)
            # print(type(predictions))

            # labels_tensor = tf.convert_to_tensor(labels_onehot, dtype=tf.float32)
            # logits_tensor = tf.convert_to_tensor(predictions, dtype=tf.float32)
            # print(type(logits_tensor))
            # labels_tensor = tf.expand_dims(labels_tensor,0)
            # print(labels_tensor.shape)
            # print(logits_tensor.shape)

            # cross_entropy = tf.losses.softmax_cross_entropy(labels_tensor, logits_tensor)
            # print(cross_entropy)
            # print(cross_entropy.eval())

            # grad = tf.gradients(cross_entropy, x)
            # print(grad)

            perb = tf.sign(grads)  # calculate perturbation
            # print(perb.eval())
            perturbation = perb.eval()
            # print(type(perturbation))
            if filter_type == 1:
                if filter_sigma > 0:
                    if filter_mode % 2 == 1:
                        if filter_dimension == 2:
                            perturbation[0] = gaussian_filter(perturbation[0], sigma=(filter_sigma, filter_sigma, 0))
                        elif filter_dimension == 3:
                            perturbation = gaussian_filter(perturbation, sigma=filter_sigma)
            elif filter_type == 2:
                if filter_mode % 2 == 1:
                    if filter_dimension == 2:
                        perturbation[0] = uniform_filter(perturbation[0], size=(3, 3, 1))
                    elif filter_dimension == 3:
                        perturbation = uniform_filter(perturbation)
            ## Test FGSM ##
            if i == int_iteration_time_N:
                original_prediction = sess.run(logits, {x: images_np, labels: label})
                adv_images = np.clip(adv_images, -0.5, 0.5)
                if filter_type == 1:
                    if filter_sigma > 0:
                        if filter_mode >= 2:
                            if filter_dimension == 2:
                                adv_images[0] = gaussian_filter(adv_images[0], sigma=(filter_sigma, filter_sigma, 0))
                            elif filter_dimension == 3:
                                adv_images = gaussian_filter(adv_images, sigma=filter_sigma)
                elif filter_type == 2:
                    if filter_mode >= 2:
                        if filter_dimension == 2:
                            adv_images[0] = uniform_filter(adv_images[0], size=(3, 3, 1))
                        elif filter_dimension == 3:
                            adv_images = uniform_filter(adv_images)
                prediction = sess.run(logits, {x: adv_images})
                if logit_type == "paper":
                    depth_map_original = feature["depth_map"]
                    masks = feature["masks"]
                    depth_map_original = depth_map_original[..., 0] * masks[..., 0]
                    depth_mean_original = np.sum(depth_map_original) / np.sum(masks[..., 0])
                    cla_ratio = flags.paras.cla_ratio
                    depth_ratio = 1 - cla_ratio
                    original_prediction[0][label_real] = (
                        depth_ratio * depth_mean_original + cla_ratio * original_prediction[0][label_real]
                    )
                    original_prediction[0][label_spoof] = 1.0 - depth_mean_original

                    depth_map_tensor_name = signature["serving_default"].outputs["depth_map"].name
                    depth_map = sess.graph.get_tensor_by_name(depth_map_tensor_name)
                    depth_map_attack = sess.run(depth_map, {x: adv_images})
                    depth_map_attack = depth_map_attack[..., 0] * masks[..., 0]
                    depth_mean_attack = np.sum(depth_map_attack) / np.sum(masks[..., 0])
                    prediction[0][label_real] = depth_ratio * depth_mean_attack + cla_ratio * prediction[0][label_real]
                    prediction[0][label_spoof] = 1.0 - depth_mean_attack
        if i < int_iteration_time_N:
            adv_images = clip(images_np * 2, adv_images * 2 - weight_alpha * perturbation, eps)
            if filter_type == 1:
                if filter_sigma > 0:
                    if filter_mode >= 2:
                        if filter_dimension == 2:
                            adv_images[0] = gaussian_filter(adv_images[0], sigma=(filter_sigma, filter_sigma, 0))
                        elif filter_dimension == 3:
                            adv_images = gaussian_filter(adv_images, sigma=filter_sigma)
            elif filter_type == 2:
                if filter_mode >= 2:
                    if filter_dimension == 2:
                        adv_images[0] = uniform_filter(adv_images[0], size=(3, 3, 1))
                    elif filter_dimension == 3:
                        adv_images = uniform_filter(adv_images)

    if flags.paras.attack_type == 1:
        if prediction[0][label_real] >= prediction[0][label_spoof]:
            print("attack success")
            n_attack_success += 1
        else:
            print("attack fail")
    elif flags.paras.attack_type == 2:
        if prediction[0][label_real] > prediction[0][label_spoof]:
            print("attack success")
            n_attack_success += 1
        else:
            print("attack fail")
    elif flags.paras.attack_type == 3:
        if prediction[0][label_real] >= 0.5:
            print("attack success")
            n_attack_success += 1
        else:
            print("attack fail")
    elif flags.paras.attack_type == 4:
        if prediction[0][label_real] > 0.5:
            print("attack success")
            n_attack_success += 1
        else:
            print("attack fail")

    # print(images_np)
    images_squeeze = np.squeeze(images_np)
    # frame1, frame2, frame3, frame4, frame5 = np.split(images_squeeze, indices_or_sections=5, axis=2)
    frames = np.split(images_squeeze, indices_or_sections=5, axis=2)
    # print(frame1.shape)
    # print(frames[0].shape)

    adv_images_squeeze = np.squeeze(adv_images)
    # adv_frame1, adv_frame2, adv_frame3, adv_frame4, adv_frame5 = np.split(adv_images_squeeze, indices_or_sections=5, axis=2)
    adv_frames = np.split(adv_images_squeeze, indices_or_sections=5, axis=2)
    # print(adv_frame1.shape)
    # print(adv_frames[0].shape)
    if not os.path.exists(os.path.join(result_path, "logits.txt")):
        os.mknod(os.path.join(result_path, "logits.txt"))

    with open(os.path.join(result_path, "logits.txt"), "a+") as f:
        if dataset == "OULU-NPU":
            f.write(f"folder: {folder_names}\toriginal: {original_prediction[0]}\tattack: {prediction[0]}\n")
        else:
            f.write(f"folder: {folder}/{folder_names}\toriginal: {original_prediction[0]}\tattack: {prediction[0]}\n")

    # if not os.path.exists(f'FGSM_result/original/{folder_names}'):
    #     os.makedirs(f'FGSM_result/original/{folder_names}')
    if dataset == "OULU-NPU":
        if not os.path.exists(f"{result_path}/attack_image/{folder_names}"):
            os.makedirs(f"{result_path}/attack_image/{folder_names}")
    else:
        if not os.path.exists(f"{result_path}/attack_image/{folder}/{folder_names}"):
            os.makedirs(f"{result_path}/attack_image/{folder}/{folder_names}")

    if not os.path.exists(os.path.join(result_path, "img_diff.txt")):
        os.mknod(os.path.join(result_path, "img_diff.txt"))

    for frame_id in range(5):
        # plt.imsave(f'FGSM_result/original/{folder_names}/frame_{frame_id + 1}.png', np.clip(frames[frame_id] + 0.5, 0, 1))
        # plt.imsave(f'{attack_type}_result/attack/{folder_names}/frame_{frame_id + 1}.png', np.clip(adv_frames[frame_id] + 0.5, 0, 1))
        img = np.clip(adv_frames[frame_id] + 0.5, 0, 1)
        img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
        if dataset == "OULU-NPU":
            img.save(f"{result_path}/attack_image/{folder_names}/frame_{frame_id + 1}.png")
        else:
            img.save(f"{result_path}/attack_image/{folder}/{folder_names}/frame_{frame_id + 1}.png")
        with open(os.path.join(result_path, "img_diff.txt"), "a+") as f:
            if dataset == "OULU-NPU":
                f.write(
                    f"folder: {folder_names}\tframe_{frame_id + 1}\timg_diff: {np.sum(adv_frames[frame_id] - frames[frame_id])}\n"
                )
            else:
                f.write(
                    f"folder: {folder}/{folder_names}\tframe_{frame_id + 1}\timg_diff: {np.sum(adv_frames[frame_id] - frames[frame_id])}\n"
                )

if not os.path.exists(os.path.join(result_path, "attack_result.txt")):
    os.mknod(os.path.join(result_path, "attack_result.txt"))
with open(os.path.join(result_path, "attack_result.txt"), "a+") as f:
    f.write(f"attack type: {flags.paras.attack_type}\n")
    f.write(f"eps: {eps}\n")
    f.write(
        f"attack success rate: {n_attack_success / (len(features_list) - n_test_real)}\tattack success: {n_attack_success}\tattack fail: {len(features_list) - n_test_real - n_attack_success}\ttest real: {n_test_real}\ttotal videos: {len(features_list)}\n"
    )
print("dataset", dataset)
print("eps", eps)
print(
    f"attack success rate: {n_attack_success / (len(features_list) - n_test_real)}\tattack success: {n_attack_success}\tattack fail: {len(features_list) - n_test_real - n_attack_success}\ttest real: {n_test_real}\ttotal videos: {len(features_list)}"
)

