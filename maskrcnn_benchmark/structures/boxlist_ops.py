# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .bounding_box import BoxList

from maskrcnn_benchmark.layers import nms as _box_nms
from maskrcnn_benchmark.layers import ml_nms as _box_ml_nms


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="score"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maxium suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def boxlist_ml_nms(boxlist, nms_thresh, max_proposals=-1,
                   score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    scores = boxlist.get_field(score_field)
    labels = boxlist.get_field(label_field)

    if boxes.device == torch.device("cpu"):
        keep = []
        unique_labels = torch.unique(labels)
        print(unique_labels)
        for j in unique_labels:
            inds = (labels == j).nonzero().view(-1)

            scores_j = scores[inds]
            boxes_j = boxes[inds, :].view(-1, 4)
            keep_j = _box_nms(boxes_j, scores_j, nms_thresh)

            keep += keep_j
    else:
        keep = _box_ml_nms(boxes, scores, labels.float(), nms_thresh)

    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]

    return boxlist.convert(mode)


def remove_small_boxes(boxlist, min_size):
    """
    Only keep boxes with both sides >= min_size

    Arguments:
        boxlist (Boxlist)
        min_size (int)
    """
    # WORK AROUND: work around unbind using split + squeeze.
    xywh_boxes = boxlist.convert("xywh").bbox
    _, _, ws, hs = xywh_boxes.split(1, dim=1)
    ws = ws.squeeze(1)
    hs = hs.squeeze(1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def boxlist_iou(boxlist1, boxlist2):
    """Compute the intersection over union of two set of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Arguments:
      box1: (BoxList) bounding boxes, sized [N,4].
      box2: (BoxList) bounding boxes, sized [M,4].

    Returns:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if boxlist1.size != boxlist2.size:
        raise RuntimeError(
            "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

    N = len(boxlist1)
    M = len(boxlist2)

    area1 = boxlist1.area()
    area2 = boxlist2.area()

    box1, box2 = boxlist1.bbox, boxlist2.bbox

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    TO_REMOVE = 1

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou


# TODO redundant, remove
def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    if isinstance(tensors[0], torch.Tensor):
        return torch.cat(tensors, dim)
    else:
        return cat_boxlist(tensors)


def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    # 收集所有字段
    all_fields = set()
    for bbox in bboxes:
        all_fields.update(bbox.fields())

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    # 对于每个字段，检查是否有bbox缺少该字段
    for field in all_fields:
        field_data = []
        for bbox in bboxes:
            if bbox.has_field(field):
                field_data.append(bbox.get_field(field))
            else:
                # 如果缺少字段，创建一个默认值
                # 尝试从其他bbox获取示例数据来确定类型
                example_value = None
                for b in bboxes:
                    if b.has_field(field):
                        example_value = b.get_field(field)
                        break

                if example_value is not None:
                    if isinstance(example_value, torch.Tensor):
                        # 创建与example_value相同类型的空张量
                        device = example_value.device
                        dtype = example_value.dtype
                        shape = (len(bbox),) + example_value.shape[1:]
                        default_value = torch.zeros(shape, dtype=dtype, device=device)
                        field_data.append(default_value)
                    elif isinstance(example_value, (int, float)):
                        # 对于标量，创建适当类型的张量
                        dtype = torch.int64 if isinstance(example_value, int) else torch.float32
                        device = next(bbox.parameters(), torch.device('cpu')).device
                        default_value = torch.zeros(len(bbox), dtype=dtype, device=device)
                        field_data.append(default_value)
                    else:
                        # 对于其他类型，跳过
                        continue
                else:
                    continue

        if field_data:
            try:
                cat_boxes.add_field(field, _cat(field_data, dim=0))
            except Exception as e:
                print(f"Warning: cannot concatenate field {field}, error: {e}")
                continue

    return cat_boxes


def getUnionBBox(aBB, bBB, margin=10):
    assert aBB.size == bBB.size
    assert aBB.mode == bBB.mode
    ih, iw = aBB.size
    union_boxes = torch.cat([(torch.min(aBB.bbox[:, [0, 1]], bBB.bbox[:, [0, 1]]) - margin).clamp(min=0), \
                             (torch.max(aBB.bbox[:, [2]], bBB.bbox[:, [2]]) + margin).clamp(max=iw), \
                             (torch.max(aBB.bbox[:, [3]], bBB.bbox[:, [3]]) + margin).clamp(max=ih)], dim=1)
    return BoxList(union_boxes, aBB.size, mode=aBB.mode)
