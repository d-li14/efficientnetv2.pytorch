import torch
import argparse
from thop import profile

from effnetv2 import *

# for mac duplicate lib bug
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def get_args_parser():
    parser = argparse.ArgumentParser('PVT training and evaluation script', add_help=False)
    # Model parameters
    parser.add_argument('-m', '--model', default='resnet50', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('-bs', '--batch_size', default=8, type=int, help='set batch size')
    parser.add_argument('-e', '--export', action='store_true', help='convert to onnx models')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('generate onnx timm models', parents=[get_args_parser()])
    args = parser.parse_args()

    model_names = ['s', 'm', 'l', 'xl', 'b0', 'b1', 'b2', 'b3']
    for m in model_names:
        model_name = "effnetv2_" + m
        model = eval(model_name)()
        print(model_name)
        x = torch.randn(1, 3, 224, 224)
        flops, params = profile(model, inputs=(x,), verbose=False)
        print("flops = %fM" % (flops / 1e6, ))
        print("param size = %fM" % (params / 1e6, ))

        if args.export:
            print("exporting....")
            model.eval()
            x = torch.randn(args.batch_size, 3, 224, 224)
            torch.onnx.export(model, x, args.model+"_bs"+str(args.batch_size)+".onnx",
                              input_names=['input'],
                              output_names=['output'],
                              verbose=True,
                              opset_version=11,
                              operator_export_type=torch.onnx.OperatorExportTypes.ONNX)
            print("exported!")
