##!/usr/bin/env python3
""" Yolo class """
import tensorflow.keras as K


class Yolo(object):
    """ Yolo """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ Initialize """
        model = K.models.load_model(model_path)
        self.model = model
        with open(classes_path, 'r') as fp:
            classes = [i.strip() for i in fp.readlines()]
            self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """ Process the outputs """
        boxes = [np.zeros(i[:, :, :, :4].shape) for i in outputs]
        box_conf = []
        box_class_prob = []
        img_height, img_width = img_size
        for i in range(len(outputs)):
            grid_height, grid_width, anchor_boxes, _ = outputs[i].shape
            t_x = outputs[i][:, :, :, 0]
            t_y = outputs[i][:, :, :, 1]
            t_w = outputs[i][:, :, :, 2]
            t_h = outputs[i][:, :, :, 3]

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            cx = np.array([np.arange(grid_width) for i in range(grid_height)])
            cx = cx.reshape(grid_width, grid_width, 1)
            cy = np.array([np.arange(grid_width) for i in range(grid_height)])
            cy = cy.reshape(grid_height, gridheight).T.reshape(grid_height,
                                                               grid_height, 1)

            bx = p_w * np.exp(t_w)
            bh = p_h * np.exp(t_h)

            boxes[i][:, :, :, 0] = (bx - (bw / 2)) * img_width
            boxes[i][:, :, :, 1] = (by - (bh / 2)) * img_height
            boxes[i][:, :, :, 2] = (bx + (bw / 2)) * img_width
            boxes[i][:, :, :, 3] = (by + (bh / 2)) * img_height

            conf = (1 / (1 + np.exp(-outputs[i][:, :, :, 4:5])))
            conf.reshape(grid_height, grid_width, anchor_boxes, 1)
            box_conf.append(conf)

            box_class = (1 / (1 + np.exp(-outputs[i][:, :, :, 5:])))
            box_class_prob.append(box_class)

        return boxes, box_conf, box_class_prob
