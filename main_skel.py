import torch
import cv2
from image_processing import preprocess
from performance import dice_coeff
from filter import create_circular_filter, circular_filter
import argparse


def main():
    parser = argparse.ArgumentParser(description="Image Skeleton")
    # Set the default for the dataset argument
    parser.add_argument("--image_path", type= str)
    parser.add_argument("--image_truth_path", type=str, default="")
    parser.add_argument("--no_circle", type= int)
    parser.add_argument("--max_radius", type=int)
    parser.add_argument("--min_radius", type=int)
    parser.add_argument("--ith_circle", type=int, default=1)
    parser.add_argument('--show', default='', action='store_true', dest='image_show', help='show result')
    parser.add_argument('--no_show', default='', action='store_false', dest='image_show', help='show result')
    parser.set_defaults(show=False)
    args = parser.parse_args()
    image_path = args.image_path
    image_truth_path = args.image_truth_path
    no_circle = args.no_circle
    min_radius = args.min_radius
    max_radius = args.max_radius
    ith_circle = args.ith_circle
    show = args.image_show

    image_input = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_process = preprocess(image_input)
    image_data = torch.tensor(image_process)
    image_data = image_data.unsqueeze(0)
    image_data = image_data.unsqueeze(0)
    filter_creation = create_circular_filter(no_circle=no_circle, min_radius=min_radius, max_radius=max_radius)
    after_filter = circular_filter(image_data, filter_creation, max_radius)
    if show:
        im_show = torch.Tensor.numpy(after_filter[0, ith_circle, :])
        show_filter_image(im_show)
    if image_truth_path != "":
        image_truth = cv2.imread(image_truth_path, cv2.IMREAD_GRAYSCALE)
        image_truth = torch.tensor(image_truth)
        image_truth = image_truth.unsqueeze(0)
        image_truth = image_truth.unsqueeze(0)
        image_truth = torch.Tensor.numpy(image_truth)
        image_filtered = torch.Tensor.numpy(after_filter[0, ith_circle, :])
        score = dice_coeff(image_truth, image_filtered)
        print("Score = ", score)


def show_filter_image(final_image):
    cv2.imshow("Result", final_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()