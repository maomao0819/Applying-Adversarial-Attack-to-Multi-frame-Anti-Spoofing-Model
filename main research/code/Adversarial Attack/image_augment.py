import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

ia.seed(1)

# Example batch of images.
# The array has shape (32, 64, 64, 3) and dtype uint8.
images = np.array(
	[ia.quokka(size=(64, 64)) for _ in range(2)],
	dtype=np.uint8
)
print("image_shape ", np.shape(images))
print("image ", images)
seq = iaa.Sequential([
	# Apply affine transformations to each image.
	# Scale/zoom them, translate/move them, rotate them and shear them.
	iaa.Affine(
		rotate=(-5, 5),
		shear=(-5, 5),
		scale={"x": (0.85, 1.15), "y": (0.85, 1.15)},
		translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}
	),

	iaa.PerspectiveTransform(scale=(0, 0.025)),

	# Make some images brighter and some darker.
	# In 20% of all cases, we sample the multiplier once per channel,
	# which can end up changing the color of the images.
	# iaa.Multiply((0.8, 1.2), per_channel=0.2),
	iaa.Multiply((0.85, 1.15)),

	# Strengthen or weaken the contrast in each image.
	iaa.LinearContrast((0.9, 1.1)),

	# Small gaussian blur with random sigma between 0 and 0.5.
	# But we only blur about 50% of all images.
	# iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
	iaa.GaussianBlur(sigma=(0, 1)),
	
	iaa.AddToHueAndSaturation((-15, 15), per_channel=True)

	# iaa.Fliplr(0.5), # horizontal flips
	# iaa.Crop(percent=(0, 0.1)), # random crops
	
	# Add gaussian noise.
	# For 50% of all images, we sample the noise once per pixel.
	# For the other 50% of all images, we sample the noise per pixel AND
	# channel. This can change the color (not only brightness) of the
	# pixels.
	# iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
		
], random_order=True) # apply augmenters in random order

images_aug = seq(images=images)
print("image_shape ", np.shape(images))
print("image ", images)