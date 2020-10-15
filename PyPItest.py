
import argparse
from skinAPI import test_model

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='PyTorch Testing')
    parser.add_argument('-img', default=r'/home/Claire/AI/testImg\nevus.jpg',
                        help='image file to test the skincare level')
    parser.add_argument('-w', default='/home/Claire/AI/skin/weights/weights.pth',
                        help='model weight to test the skincare level')

    args = parser.parse_args()
    test_model(args.img, args.w)

