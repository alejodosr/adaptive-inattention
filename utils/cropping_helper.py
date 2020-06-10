import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np
from math import ceil,floor

import cv2
import math

def check_same(stride):
    if isinstance(stride, (list, tuple)):
        assert len(stride) == 2 and stride[0] == stride[1]
        stride = stride[0]
    return stride

def receptive_field(model, input_size, batch_size=-1, device="cuda"):
    '''
    :parameter
    'input_size': tuple of (Channel, Height, Width)
    :return  OrderedDict of `Layername`->OrderedDict of receptive field stats {'j':,'r':,'start':,'conv_stage':,'output_shape':,}
    'j' for "jump" denotes how many pixels do the receptive fields of spatially neighboring units in the feature tensor
        do not overlap in one direction.
        i.e. shift one unit in this feature map == how many pixels shift in the input image in one direction.
    'r' for "receptive_field" is the spatial range of the receptive field in one direction.
    'start' denotes the center of the receptive field for the first unit (start) in on direction of the feature tensor.
        Convention is to use half a pixel as the center for a range. center for `slice(0,5)` is 2.5.
    '''
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(receptive_field)
            m_key = "%i" % module_idx
            p_key = "%i" % (module_idx - 1)
            receptive_field[m_key] = OrderedDict()

            if not receptive_field["0"]["conv_stage"]:
                ##print("Enter in deconv_stage")
                receptive_field[m_key]["j"] = 0
                receptive_field[m_key]["r"] = 0
                receptive_field[m_key]["start"] = 0
            else:
                p_j = receptive_field[p_key]["j"]
                p_r = receptive_field[p_key]["r"]
                p_start = receptive_field[p_key]["start"]
                
                if class_name == "Conv2d" or class_name == "MaxPool2d":
                    kernel_size = module.kernel_size
                    stride = module.stride
                    padding = module.padding
                    kernel_size, stride, padding = map(check_same, [kernel_size, stride, padding])
                    receptive_field[m_key]["j"] = p_j * stride
                    receptive_field[m_key]["r"] = p_r + (kernel_size - 1) * p_j
                    receptive_field[m_key]["start"] = p_start + ((kernel_size - 1) / 2 - padding) * p_j
                elif class_name == "BatchNorm2d" or class_name == "ReLU" or class_name == "Bottleneck" or class_name == "ReLU6":
                    receptive_field[m_key]["j"] = p_j
                    receptive_field[m_key]["r"] = p_r
                    receptive_field[m_key]["start"] = p_start
                elif class_name == "ConvTranspose2d":
                    receptive_field["0"]["conv_stage"] = False
                    receptive_field[m_key]["j"] = 0
                    receptive_field[m_key]["r"] = 0
                    receptive_field[m_key]["start"] = 0
                else:
                    #print(class_name)
                    raise ValueError("module not ok")
                    pass
            receptive_field[m_key]["input_shape"] = list(input[0].size()) # only one
            receptive_field[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                # list/tuple
                receptive_field[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                # tensor
                receptive_field[m_key]["output_shape"] = list(output.size())
                receptive_field[m_key]["output_shape"][0] = batch_size

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [Variable(torch.rand(2, *in_size)).type(dtype) for in_size in input_size]
    else:
        x = Variable(torch.rand(2, *input_size)).type(dtype)

    # create properties
    receptive_field = OrderedDict()
    receptive_field["0"] = OrderedDict()
    receptive_field["0"]["j"] = 1.0
    receptive_field["0"]["r"] = 1.0
    receptive_field["0"]["start"] = 0.5
    receptive_field["0"]["conv_stage"] = True
    receptive_field["0"]["output_shape"] = list(x.size())
    receptive_field["0"]["output_shape"][0] = batch_size
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    #print("------------------------------------------------------------------------------")
    line_new = "{:>20}  {:>10} {:>10} {:>10} {:>15} ".format("Layer (type)", "map size", "start", "jump", "receptive_field")
    #print(line_new)
    #print("==============================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in receptive_field:
        # input_shape, output_shape, trainable, nb_params
        assert "start" in receptive_field[layer], layer
        assert len(receptive_field[layer]["output_shape"]) == 4
        line_new = "{:7} {:12}  {:>10} {:>10} {:>10} {:>15} ".format(
            "",
            layer,
            str(receptive_field[layer]["output_shape"][2:]),
            str(receptive_field[layer]["start"]),
            str(receptive_field[layer]["j"]),
            format(str(receptive_field[layer]["r"]))
        )
        #print(line_new)

    #print("==============================================================================")
    # add input_shape
    receptive_field["input_size"] = input_size
    return receptive_field


def receptive_field_for_unit(receptive_field_dict, layer, unit_position):
    """Utility function to calculate the receptive field for a specific unit in a layer
        using the dictionary calculated above
    :parameter
        'layer': layer name, should be a key in the result dictionary
        'unit_position': spatial coordinate of the unit (H, W)
    ```
    alexnet = models.alexnet()
    model = alexnet.features.to('cuda')
    receptive_field_dict = receptive_field(model, (3, 224, 224))
    receptive_field_for_unit(receptive_field_dict, "8", (6,6))
    ```
    Out: [(62.0, 161.0), (62.0, 161.0)]
    """
    input_shape = receptive_field_dict["input_size"]
    if layer in receptive_field_dict:
        rf_stats = receptive_field_dict[layer]
        assert len(unit_position) == 2
        feat_map_lim = rf_stats['output_shape'][2:]
        if np.any([unit_position[idx] < 0 or
                   unit_position[idx] >= feat_map_lim[idx]
                   for idx in range(2)]):
            raise Exception("Unit position outside spatial extent of the feature tensor ((H, W) = (%d, %d)) " % tuple(feat_map_lim))
        # X, Y = tuple(unit_position)
        rf_range = [(rf_stats['start'] + idx * rf_stats['j'] - rf_stats['r'] / 2,
            rf_stats['start'] + idx * rf_stats['j'] + rf_stats['r'] / 2) for idx in unit_position]
        if len(input_shape) == 2:
            limit = input_shape
        else:  # input shape is (channel, H, W)
            limit = input_shape[1:3]
        rf_range = [(max(0, rf_range[axis][0]), min(limit[axis], rf_range[axis][1])) for axis in range(2)]
        #print("Receptive field size for layer %s, unit_position %s,  is \n %s" % (layer, unit_position, rf_range))
        return rf_range
    else:
        raise KeyError("Layer name incorrect, or not included in the model.")



def adjust_bbox(bbox,cropping_overlap,image_w,image_h,model, use_hardcoded=True):
    #BEGIN
    #ALL COORDINATES ARE TOPLEFT CORNER
    output={}
    if(use_hardcoded):
        last_layer_dict={'j':16.0, 'r':219.0}
    else:
        parsed_model = receptive_field(model, (3,image_w,image_h))
        last_layer_key =   list(parsed_model.keys())[-2] #last is input size 
        last_layer_dict=parsed_model[last_layer_key]
        print('=======================================')
        print('HARDCODE THIS VALUES TO last_layer_dict')
        print('j: '+ str(last_layer_dict['j']))
        print('r: '+ str(last_layer_dict['r']))
        print('=======================================')
    input_stride = last_layer_dict['j']
    (x,y,w,h)= bbox
    if(x<0 or y<0 or w<=0 or h<=0 or x+w >image_w or y+h>image_h):
        raise Exception('Not valid bbox : '+ str(bbox))
    min_x = adjust2grid(x, input_stride,down=True)
    min_y = adjust2grid(y, input_stride,down=True)
    max_x = adjust2grid(x+w, input_stride,down=False)
    max_y = adjust2grid(y+h, input_stride,down=False)

    x=min_x
    y=min_y
    w=max_x-min_x
    h=max_y-min_y
    output['input_bbox'] = ( x,y,w,h)
    output['output_bbox'] = ( x/input_stride,y/input_stride,w/input_stride,h/input_stride)
    output['cut_x_output']=(0,0)
    output['cut_y_output']=(0,0)
    # if(cropping_overlap>0):
    cell_extension = ceil(ceil((last_layer_dict['r'] /2.0 ) * cropping_overlap )/input_stride)
    min_x_cells= min(min_x /input_stride, cell_extension)
    min_y_cells=min(min_y/input_stride, cell_extension)
    max_x_cells = min((image_w/input_stride)-(max_x /input_stride), cell_extension)
    max_y_cells =min((image_h/input_stride)-(max_y /input_stride), cell_extension)
    output['cut_x_output']=(min_x_cells,max_x_cells)
    output['cut_y_output']=(min_y_cells,max_y_cells)
    min_x = min_x - min_x_cells * input_stride
    min_y = min_y - min_y_cells * input_stride
    max_x = max_x + max_x_cells * input_stride
    max_y = max_y + max_y_cells * input_stride
    x=min_x
    y=min_y
    w=max_x-min_x
    h=max_y-min_y
    output['input_bbox'] = ( x,y,w,h)
    return(output)
def adjust2grid( coordinate, grid_size, down):
    coordinate= float(coordinate)/float(grid_size)
    if(down):
        coordinate=floor(coordinate)
    else:
        coordinate=ceil(coordinate)
    coordinate= coordinate*grid_size
    return coordinate



    #USAGE
#     bbox= ( 50,12,10,15) # ( x,y,w,h)
# cropping_overlap = 0.5
# image_w=320
# image_h=320
# model = models.mvod_basenet.MobileNetV1(num_classes=2, alpha=2).cuda()
# print(adjust_bbox(bbox,cropping_overlap,image_w,image_h,model))


def process_intermmediate_fmap(fmap, window_name='fmap_img_draw', cropped_fmap=None, adjustment_dict=None,
                               plot_image=True):
    # Store fmap
    faked_fmap = fmap.clone()

    if cropped_fmap is not None and adjustment_dict is not None:
        # Rescale coordinates
        x1 = int(adjustment_dict['output_bbox'][0])
        y1 = int(adjustment_dict['output_bbox'][1])
        w = int(adjustment_dict['output_bbox'][2])
        h = int(adjustment_dict['output_bbox'][3])
        adjusted_feature_map = cropped_fmap[:, :, int(adjustment_dict['cut_y_output'][0]):int(cropped_fmap.shape[2]) - int(adjustment_dict['cut_y_output'][1]),
                               int(adjustment_dict['cut_x_output'][0]):int(cropped_fmap.shape[3]) - int(adjustment_dict['cut_x_output'][1])]
        # adjusted_feature_map = torch.zeros([1, 512, h, w], dtype=torch.int32)

        if plot_image:
            cropped_fmap_img = adjusted_feature_map.squeeze(0).cpu().numpy().astype(np.float32).transpose((1, 2, 0))

        faked_fmap[:, :, y1:y1+h, x1:x1+w] = adjusted_feature_map.clone()

    if plot_image:
        # Convert to opencv image
        fmap_img = fmap.squeeze(0).cpu().numpy().astype(np.float32).transpose((1, 2, 0))
        square_size = int(math.sqrt(fmap_img.shape[2]))
        index = 0
        channel_w, channel_h, channel_depth = fmap_img.shape

        fmap_img_draw = np.zeros((square_size * channel_w, square_size * channel_h, 3))

        channel_img_rgb = np.zeros((channel_w, channel_h, 3), dtype="uint8")
        for i in range(square_size):
            for j in range(square_size):
                # Get channel
                channel_img = cv2.normalize(fmap_img[:, :, index], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                channel_img_rgb[:, :, 0] = channel_img
                channel_img_rgb[:, :, 1] = channel_img
                channel_img_rgb[:, :, 2] = channel_img

                if cropped_fmap is not None and adjustment_dict is not None:
                    # Introduce cropped channel
                    cropped_channel_img = cv2.normalize(cropped_fmap_img[:, :, index], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
                    channel_img_rgb[y1:y1 + h, x1:x1 + w, 0] = cropped_channel_img


                # Include channel in image to draw
                fmap_img_draw[int(i * channel_w):int(i * channel_w + channel_w),
                                int(j * channel_h):int(j * channel_h + channel_h)] = channel_img_rgb

                # Increase index
                index += 1

        # Show image
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 800)
        cv2.imshow(window_name, fmap_img_draw)

    return faked_fmap

def visualize_fmap(fmap, window_name='fmap'):
    # Convert to opencv image
    fmap_img = fmap.cpu().numpy().astype(np.float32)
    square_size = int(math.sqrt(fmap_img.shape[2]))
    index = 0
    channel_w, channel_h, channel_depth = fmap_img.shape

    fmap_img_draw = np.zeros((square_size * channel_w, square_size * channel_h, 3))

    channel_img_rgb = np.zeros((channel_w, channel_h, 3), dtype="uint8")
    for i in range(square_size):
        for j in range(square_size):
            # Get channel
            channel_img = cv2.UMat(fmap_img[:, :, index]).get()  # Found on github due to errors

            # channel_img = cv2.normalize(fmap_img[:, :, index], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            channel_img_rgb[:, :, 0] = channel_img
            channel_img_rgb[:, :, 1] = channel_img
            channel_img_rgb[:, :, 2] = channel_img

            # Include channel in image to draw
            fmap_img_draw[int(i * channel_w):int(i * channel_w + channel_w),
                            int(j * channel_h):int(j * channel_h + channel_h)] = channel_img_rgb

            # Increase index
            index += 1

        # Show image
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 500, 500)
        cv2.imshow(window_name, fmap_img_draw)