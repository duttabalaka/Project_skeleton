import torch
import cv2
from image_processing import preprocess, binarize
from performance import dice_coeff
from filter import create_circular_filter, circular_filter, render_radial_skeleton
import argparse
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Image Skeleton")
    # Set the default for the dataset argument
    parser.add_argument("--image_path", type= str)
    parser.add_argument("--image_truth_path", type=str, default="")
    #parser.add_argument("--no_circle", type= int)
    parser.add_argument("--max_radius", type=int)
    #parser.add_argument("--min_radius", type=int)
    #parser.add_argument("--ith_circle", type=int, default=1)
    parser.add_argument('--show', default='', action='store_true', dest='image_show', help='show result')
    parser.add_argument('--no_show', default='', action='store_false', dest='image_show', help='show result')
    parser.set_defaults(show=False)
    args = parser.parse_args()
    image_path = args.image_path
    image_truth_path = args.image_truth_path
    no_circle = args.max_radius + 1
    min_radius = 0
    max_radius = args.max_radius
    #ith_circle = args.ith_circle
    show = args.image_show

    image_input = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radial_skeleton = preprocess(image_input)


    image_data = torch.tensor(radial_skeleton)
    image_data = image_data.unsqueeze(0)
    image_data = image_data.unsqueeze(0)
    filter = create_circular_filter(no_circle=no_circle, min_radius=0, max_radius=no_circle-1)
    channel_radii_images = circular_filter(torch.clamp(image_data,0,1), filter, max_radius).cpu().numpy()
    circle_bin = render_radial_skeleton(channel_radii_images,radial_skeleton,filter)
    fig, arr = plt.subplots(2,3)
    gt_bin = cv2.imread(image_truth_path, cv2.IMREAD_GRAYSCALE)==0
    otsu_bin = binarize(image_input)
    #circle_bin = channel_radii_images[0, ith_circle, :]>0
    print("Otsu Dice Score:", dice_coeff(gt_bin, otsu_bin))
    print("Circle Dice Score:", dice_coeff(gt_bin, circle_bin))
    if show:
        arr[0][0].imshow(otsu_bin)
        arr[1][0].imshow(circle_bin)
        arr[0][1].imshow(radial_skeleton)
        arr[1][1].imshow(channel_radii_images[0, -1, :,:])
        arr[0][2].imshow(otsu_bin!=gt_bin)
        arr[1][2].imshow(circle_bin!=gt_bin)

    #print("After range:" ,channel_radii_images[0, ith_circle, :].cpu().numpy().max(),
    #      channel_radii_images[0, ith_circle, :].cpu().numpy().min(), (channel_radii_images[0, ith_circle, :].cpu()>0).numpy().mean())
        plt.show()
    if image_truth_path != "":
        gt_bin = cv2.imread(image_truth_path, cv2.IMREAD_GRAYSCALE)
        gt_bin = torch.tensor(gt_bin)
        gt_bin = gt_bin.unsqueeze(0)
        gt_bin = gt_bin.unsqueeze(0)
        gt_bin = torch.Tensor.numpy(gt_bin)
        #image_filtered = torch.Tensor.numpy(channel_radii_images[0, ith_circle, :])
        #score = dice_coeff(gt_bin, image_filtered)
        #print("Score = ", score)


def show_filter_image(final_image, caption="Caption"):
    cv2.imshow(caption, final_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
    cv2.waitKey(0)