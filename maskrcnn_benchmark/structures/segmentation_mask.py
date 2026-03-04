# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

import pycocotools.mask as mask_utils

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Mask(object):
    """
    This class is unfinished and not meant for use yet
    It is supposed to contain the mask for an object as
    a 2d tensor
    """

    def __init__(self, masks, size, mode):
        self.masks = masks
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 2
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        flip_idx = list(range(dim)[::-1])
        flipped_masks = self.masks.index_select(dim, flip_idx)
        return Mask(flipped_masks, self.size, self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]

        cropped_masks = self.masks[:, box[1]: box[3], box[0]: box[2]]
        return Mask(cropped_masks, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        pass


class Polygons(object):
    """
    This class holds a set of polygons that represents a single instance
    of an object mask. The object can be represented as a set of
    polygons
    """

    def __init__(self, polygons, size, mode):
        # assert isinstance(polygons, list), '{}'.format(polygons)
        if isinstance(polygons, list):
            # 过滤掉字符串类型的元素
            filtered_polygons = []
            for p in polygons:
                if isinstance(p, str):
                    print(f"Warning: skipping string in polygons: {p}")
                    continue
                try:
                    filtered_polygons.append(torch.as_tensor(p, dtype=torch.float32))
                except Exception as e:
                    print(f"Warning: cannot convert {type(p)} to tensor: {e}")
                    continue
            polygons = filtered_polygons
        elif isinstance(polygons, Polygons):
            polygons = polygons.polygons

        self.polygons = polygons
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped_polygons = []
        width, height = self.size
        if method == FLIP_LEFT_RIGHT:
            dim = width
            idx = 0
        elif method == FLIP_TOP_BOTTOM:
            dim = height
            idx = 1

        for poly in self.polygons:
            p = poly.clone()
            TO_REMOVE = 1
            p[idx::2] = dim - poly[idx::2] - TO_REMOVE
            flipped_polygons.append(p)

        return Polygons(flipped_polygons, size=self.size, mode=self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]

        # TODO chck if necessary
        w = max(w, 1)
        h = max(h, 1)

        cropped_polygons = []
        for poly in self.polygons:
            p = poly.clone()
            p[0::2] = p[0::2] - box[0]  # .clamp(min=0, max=w)
            p[1::2] = p[1::2] - box[1]  # .clamp(min=0, max=h)
            cropped_polygons.append(p)

        return Polygons(cropped_polygons, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled_polys = []
            for p in self.polygons:
                # 使用递归函数处理任意深度的嵌套结构
                def scale_value(val):
                    if val is None:
                        return val
                    elif isinstance(val, str):
                        # 如果是字符串，跳过并打印警告
                        print(f"Warning: skipping string in resize: {val}")
                        return val
                    elif isinstance(val, torch.Tensor):
                        return val * ratio
                    elif isinstance(val, (list, tuple)):
                        return [scale_value(v) for v in val]
                    elif isinstance(val, (int, float)):
                        return val * ratio
                    else:
                        # 如果是其他类型，尝试直接乘法
                        try:
                            return val * ratio
                        except:
                            print(f"Warning: cannot multiply {type(val)} by ratio")
                            return val

                scaled_val = scale_value(p)
                if scaled_val is not None:
                    scaled_polys.append(scaled_val)
            return Polygons(scaled_polys, size, mode=self.mode)

        ratio_w, ratio_h = ratios
        scaled_polygons = []
        for poly in self.polygons:
            # 使用递归函数处理不等比例缩放
            def scale_xy(val, depth=0):
                if val is None:
                    return val
                elif isinstance(val, str):
                    print(f"Warning: skipping string in resize: {val}")
                    return val
                elif isinstance(val, torch.Tensor):
                    result = val.clone()
                    result[0::2] *= ratio_w
                    result[1::2] *= ratio_h
                    return result
                elif isinstance(val, (list, tuple)):
                    result = []
                    for i, v in enumerate(val):
                        # 根据深度和索引判断是x坐标还是y坐标
                        if depth == 0:
                            # 第一层，按照索引奇偶判断
                            if i % 2 == 0:
                                result.append(v * ratio_w)
                            else:
                                result.append(v * ratio_h)
                        else:
                            # 更深层次，递归处理
                            result.append(scale_xy(v, depth + 1))
                    return result
                elif isinstance(val, (int, float)):
                    # 如果是单个数字，无法确定是x还是y，返回原值
                    return val
                else:
                    return val

            scaled_val = scale_xy(poly, 0)
            if scaled_val is not None:
                scaled_polygons.append(scaled_val)

        return Polygons(scaled_polygons, size=size, mode=self.mode)

    def convert(self, mode):
        width, height = self.size
        if mode == "mask":
            # 确保所有多边形都是 tensor 格式
            polys_for_conversion = []
            for p in self.polygons:
                if p is None:
                    continue
                elif isinstance(p, str):
                    print(f"Warning: skipping string in convert: {p}")
                    continue
                elif isinstance(p, torch.Tensor):
                    polys_for_conversion.append(p.detach().numpy())
                elif isinstance(p, (list, tuple)):
                    polys_for_conversion.append(p)
                else:
                    polys_for_conversion.append(p)

            if not polys_for_conversion:
                return torch.zeros((height, width), dtype=torch.uint8)

            rles = mask_utils.frPyObjects(
                polys_for_conversion, height, width
            )
            rle = mask_utils.merge(rles)
            mask = mask_utils.decode(rle)
            mask = torch.from_numpy(mask)
            # TODO add squeeze?
            return mask

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_polygons={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s


class SegmentationMask(object):
    """
    This class stores the segmentations for all objects in the image
    """

    def __init__(self, polygons, size, mode=None):
        """
        Arguments:
            polygons: a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.
        """
        assert isinstance(polygons, list)

        self.polygons = [Polygons(p, size, mode) for p in polygons if p is not None and not isinstance(p, str)]
        self.size = size
        self.mode = mode

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented"
            )

        flipped = []
        for polygon in self.polygons:
            flipped.append(polygon.transpose(method))
        return SegmentationMask(flipped, size=self.size, mode=self.mode)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        cropped = []
        for polygon in self.polygons:
            cropped.append(polygon.crop(box))
        return SegmentationMask(cropped, size=(w, h), mode=self.mode)

    def resize(self, size, *args, **kwargs):
        scaled = []
        for polygon in self.polygons:
            scaled.append(polygon.resize(size, *args, **kwargs))
        return SegmentationMask(scaled, size=size, mode=self.mode)

    def to(self, *args, **kwargs):
        return self

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_polygons = [self.polygons[item]]
        else:
            # advanced indexing on a single dimension
            selected_polygons = []
            if isinstance(item, torch.Tensor) and item.dtype == torch.bool:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.polygons[i])
        return SegmentationMask(selected_polygons, size=self.size, mode=self.mode)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s
