import numpy as np
import cv2
import os

def enhance_image(img_path, output_dir):
    # ... (existing code)
    img_path = 'lion.jpeg'
    img = cv2.imread(img_path)

    # Colorize the grayscale image
    proto_file = 'colorization_deploy_v2.prototxt'
    model_file = 'colorization_release_v2.caffemodel'
    hull_pts = 'pts_in_hull.npy'

    net = cv2.dnn.readNetFromCaffe(proto_file, model_file)
    kernel = np.load(hull_pts)


    scaled = img.astype("float32") / 255.0
    lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    resized = cv2.resize(lab_img, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))

    L = cv2.split(lab_img)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    def enhance_color_image(output_image, contrast=1.2, brightness=0.5, saturation=1.5):
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(output_image, cv2.COLOR_BGR2LAB)
        # Split the LAB image into L, A, and B channels
        l, a, b = cv2.split(lab)

        # Apply contrast enhancement
        enhanced_l = cv2.convertScaleAbs(l, alpha=contrast, beta=0)

        # Apply brightness adjustment
        enhanced_l = np.clip(enhanced_l + brightness, 0, 255).astype(np.uint8)

        # Apply saturation adjustment
        enhanced_lab = cv2.merge([enhanced_l, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        enhanced_s = cv2.convertScaleAbs(s, alpha=saturation, beta=0)
        enhanced_hsv = cv2.merge([h, enhanced_s, v])
        enhanced_colorized = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

        return enhanced_colorized

    enhanced_output_image = enhance_color_image(colorized)
    # Perform colorization and enhancement

    # Save colorized image
    colorized_output_path = os.path.join(output_dir, 'colorized', 'colorized_output.jpg')
    cv2.imwrite(colorized_output_path, colorized)

    # Perform color enhancement
    enhanced_output_image = enhance_color_image(colorized)
    
    # Save enhanced image
    enhanced_output_path = os.path.join(output_dir, 'enhanced', 'enhanced_output.jpg')
    cv2.imwrite(enhanced_output_path, enhanced_output_image)

    return enhanced_output_image

if __name__ == "__main__":
    img_path = 'BRIDGE.jpeg'
    output_dir = 'static'  # Directory where output images will be saved

    # Create output directories if they don't exist
    os.makedirs(os.path.join(output_dir, 'colorized'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'enhanced'), exist_ok=True)

    enhanced_output_image = enhance_image(img_path, output_dir)
