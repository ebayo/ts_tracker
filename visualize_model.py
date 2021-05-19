import argparse
import ts_utils.yolov5_bridge as yolo
from torchviz import make_dot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=str,
                        help='.pt file of a trained model')
    parser.add_argument('image_path', type=str,
                        help='path to database image folder')
    parser.add_argument('im_size', type=int,
                        help='Image size the model has been trained on')

    param = parser.parse_args()

    hyper = dict(conf_th=0.25, iou_th=0.5, device='cpu')

    yolo_model = yolo.Yolo5Model(param.weights, hyper, image_size=param.im_size)
    image_loader = yolo.ImageLoader(param.image_path, img_size=param.im_size)

    flag = False
    for _, img in image_loader:
        if img is None:
            pass
        if flag:
            break
        else:
            out = yolo_model.inference(img)
            print('pred \n')
            print(out[0].shape)
            par = yolo_model.get_named_params()
            print('parameters \n')

            d_par = dict(par)
            print(len(d_par))
            #dot = make_dot(out, params=d_par)
            dot = make_dot(out)
            print(dot)
            dot.format = 'png'
            dot.render("results/dot")
            flag = True
